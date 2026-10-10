import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { AttachmentId, AttachmentStore } from '@deepseek-ai/dsh-attachment'
import type { ImageAttachmentRef, SaveImageAttachment, StoredImageAttachment } from '@deepseek-ai/dsh-attachment'
import type { AssistantMessage, AssistantMessageEvent } from '@earendil-works/pi-ai'
import { GeneratedImageOutput } from '../src/generated-images.ts'
import { BlockAssembler } from '@deepseek-ai/dsh-llm'
import { toStreamChunks } from '../src/stream.ts'

const ref: ImageAttachmentRef = {
  attachmentId: AttachmentId(`sha256:${'a'.repeat(64)}`), mediaType: 'image/png', bytes: 1, width: 1, height: 1,
}

class Store extends AttachmentStore {
  imageLimits = { maxImageBytes: 3, maxImagesPerMessage: 2, maxMessageImageBytes: 6, maxImagePixels: 1, maxImageDimension: 1, mediaTypes: ['image/png', 'image/jpeg', 'image/webp'] as const }
  saved: SaveImageAttachment[] = []
  async validateImage(): Promise<void> {}
  async saveImage(input: SaveImageAttachment): Promise<ImageAttachmentRef> { this.saved.push(input); return ref }
  async readImage(): Promise<StoredImageAttachment> { throw new Error('not used') }
}

const contexts: Context[] = []
afterEach(async () => { await Promise.all(contexts.splice(0).map(ctx => ctx.fiber.dispose())) })

async function setup(path: string | undefined = '/model/generated image.png'): Promise<{ images: GeneratedImageOutput; store: Store; abort: AbortController }> {
  const ctx = new Context()
  contexts.push(ctx)
  const store = new Store(ctx)
  const abort = new AbortController()
  return {
    store, abort,
    images: new GeneratedImageOutput(() => store, () => path === undefined ? undefined : { readonlyPath: path }, abort.signal),
  }
}

function completed(overrides: Record<string, unknown> = {}): unknown {
  return { type: 'response.completed', response: { output: [{ type: 'image_generation_call', status: 'completed', result: 'YQ==', ...overrides }] } }
}

function message(content: AssistantMessage['content'] = [{ type: 'text', text: '' }]): AssistantMessage {
  return {
    role: 'assistant', api: 'openai-responses', provider: 'gateway', model: 'm', timestamp: 0,
    content, stopReason: content.some(block => block.type === 'toolCall') ? 'toolUse' : 'stop',
    usage: {
      input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
  }
}

async function* done(value = message()): AsyncGenerator<AssistantMessageEvent> {
  for (const [contentIndex, block] of value.content.entries()) {
    if (block.type === 'text') {
      yield { type: 'text_start', contentIndex, partial: value }
      yield { type: 'text_end', contentIndex, content: block.text, partial: value }
    }
  }
  yield { type: 'done', reason: value.stopReason as 'stop' | 'toolUse', message: value }
}

async function collect(events: AsyncIterable<AssistantMessageEvent>): Promise<AssistantMessageEvent[]> {
  const result: AssistantMessageEvent[] = []
  for await (const event of events) result.push(event)
  return result
}

describe('Responses hosted image output', () => {
  it('admits terminal-only images and keeps text links aligned with replay', async () => {
    const { images, store } = await setup()
    await images.observe({ type: 'response.output_item.done', item: { type: 'image_generation_call', result: 'YQ==' } })
    expect(store.saved).toEqual([])
    await images.observe(completed())
    const original = message()
    const events = await collect(images.events(done(original)))
    expect(original.content).toEqual([{ type: 'text', text: '' }])
    expect(store.saved[0]?.data).toEqual(new Uint8Array([97]))
    const assembler = new BlockAssembler()
    for await (const chunk of toStreamChunks((async function* () { yield* events })())) assembler.push(chunk)
    expect(assembler.blocks()).toEqual([{ type: 'text', text: '' }, { type: 'text', text: '![Generated image](</model/generated image.png>)' }])
    expect(assembler.finish).toEqual({ kind: 'stop' })
    expect(assembler.replayState?.blocks).toEqual([{ type: 'text' }, { type: 'text' }])
  })

  it('preserves function calls in mixed image/function responses for the agent loop', async () => {
    const { images } = await setup()
    await images.observe(completed())
    const call = { type: 'toolCall' as const, id: 'call_1|fc_1', name: 'echo', arguments: { text: 'hi' } }
    const terminal = (await collect(images.events(done(message([call]))))).at(-1)
    expect(terminal).toMatchObject({ type: 'done', reason: 'toolUse', message: { stopReason: 'toolUse', content: [call, { type: 'text' }] } })
  })

  it('does not change text-only or unsuccessful streams', async () => {
    const { images } = await setup()
    for (const event of [null, [], {}, { type: 'response.completed', response: null }, { type: 'response.completed', response: {} }, { type: 'response.completed', response: { output: [null, { type: 'message' }] } }]) {
      await images.observe(event)
    }
    const value = message([{ type: 'text', text: 'hello' }])
    expect(await collect(images.events(done(value)))).toEqual(await collect(done(value)))
    await images.observe(completed())
    const error = { ...value, stopReason: 'error' as const }
    const event: AssistantMessageEvent = { type: 'error', reason: 'error', error }
    expect(await collect(images.events((async function* () { yield event })()))).toEqual([event])
  })

  it.each([['png', 'image/png'], [null, 'image/png'], ['jpeg', 'image/jpeg'], ['webp', 'image/webp']] as const)('admits %s', async (format, type) => {
    const { images, store } = await setup('/model/100%<preview>?#.png')
    await images.observe(completed({ output_format: format }))
    expect(store.saved[0]?.mediaType).toBe(type)
    expect(await collect(images.events(done()))).toContainEqual(expect.objectContaining({ type: 'text_delta', delta: '![Generated image](</model/100%25%3Cpreview%3E%3F%23.png>)' }))
  })

  it('encodes Windows separators and refuses paths the file renderer cannot represent', async () => {
    const { images } = await setup('C:\\images\\generated.png')
    await images.observe(completed())
    expect(await collect(images.events(done()))).toContainEqual(expect.objectContaining({ type: 'text_delta', delta: '![Generated image](<C:%5Cimages%5Cgenerated.png>)' }))
    const { images: invalid } = await setup('/model/new\nline.png')
    await expect(invalid.observe(completed())).rejects.toThrow(/control characters/)
  })

  it.each([
    [{ status: 'failed' }, /without image data/],
    [{ result: '' }, /without image data/],
    [{ result: 1 }, /without image data/],
    [{ result: 'YWJjZA==' }, /byte limit/],
    [{ output_format: 'gif' }, /unsupported output format/],
    [{ result: 'not base64' }, /byte limit/],
    [{ result: 'a' }, /canonical base64/],
  ])('refuses malformed or oversized output %j', async (item, error) => {
    const { images, store } = await setup()
    await expect(images.observe(completed(item))).rejects.toThrow(error)
    expect(store.saved).toEqual([])
  })

  it('requires attachment storage and filesystem access', async () => {
    const missing = new GeneratedImageOutput(undefined, undefined, new AbortController().signal)
    await expect(missing.observe(completed())).rejects.toThrow(/attachment service/)
    const { store, abort } = await setup()
    const inaccessible = new GeneratedImageOutput(() => store, undefined, abort.signal)
    await expect(inaccessible.observe(completed())).rejects.toThrow(/readable attachment path/)
    const unmapped = new GeneratedImageOutput(() => store, () => undefined, abort.signal)
    await expect(unmapped.observe(completed())).rejects.toThrow(/readable attachment path/)
  })

  it('checks batch bounds before decoding or saving', async () => {
    const { images, store } = await setup()
    const item = { type: 'image_generation_call', status: 'completed', result: 'YWJj' }
    await expect(images.observe({ type: 'response.completed', response: { output: [item, item, item] } })).rejects.toThrow(/count limit/)
    store.imageLimits.maxImageBytes = 30
    store.imageLimits.maxMessageImageBytes = 1
    await expect(images.observe({ type: 'response.completed', response: { output: [{ ...item, result: 'YWJjZGVmZ2hpamts' }] } })).rejects.toThrow(/batch byte limit/)
    expect(store.saved).toEqual([])
  })

  it('honors cancellation before admission and before publishing links', async () => {
    const { images, store, abort } = await setup()
    abort.abort(new Error('cancelled'))
    await expect(images.observe(completed())).rejects.toThrow('cancelled')
    expect(store.saved).toEqual([])
    const second = new AbortController()
    const output = new GeneratedImageOutput(() => store, () => { throw new Error('must not publish') }, second.signal)
    store.saveImage = async () => { second.abort(new Error('cancelled after save')); return ref }
    await expect(output.observe(completed())).rejects.toThrow('cancelled after save')
  })
})
