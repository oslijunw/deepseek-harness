/** Completed Responses image output, stored before assistant settlement. @module dsh-llm-pi-ai/generated-images */

import { admitEncodedImages } from '@deepseek-ai/dsh-attachment'
import type { AttachmentStore, EncodedImageAttachment, ImageAttachmentRef, ImageMediaType } from '@deepseek-ai/dsh-attachment'
import { LlmError } from '@deepseek-ai/dsh-llm'
import type { ImageAttachmentAccess } from '@deepseek-ai/dsh-llm'
import type { AssistantMessageEvent, TextContent } from '@earendil-works/pi-ai'

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function mediaType(format: unknown): ImageMediaType {
  switch (format) {
    case undefined:
    case null:
    case 'png': return 'image/png'
    case 'jpeg': return 'image/jpeg'
    case 'webp': return 'image/webp'
    default: throw new LlmError('Responses image generation returned an unsupported output format', 'UNSUPPORTED_CONTENT')
  }
}

/**
 * One request's hosted image results. Completed images become ordinary text
 * blocks containing local Markdown links; pi-ai replay keeps those same blocks.
 * Hosted calls have already executed and must not become client function calls.
 */
export class GeneratedImageOutput {
  private content: TextContent[] = []

  constructor(
    private readonly resolveAttachments: (() => AttachmentStore | undefined) | undefined,
    private readonly resolveAccess: ((
      attachments: AttachmentStore, ref: ImageAttachmentRef,
    ) => ImageAttachmentAccess | undefined) | undefined,
    private readonly signal: AbortSignal,
  ) {}

  /**
   * Observe the authoritative completed response, including terminal-only
   * results. Admission enforces the attachment store's existing image limits.
   * Failure aborts generation; saved objects are not rolled back. Only a
   * successful pi-ai terminal event publishes their links to the session.
   * @param data - read-only raw provider event.
   */
  async observe(data: unknown): Promise<void> {
    if (!record(data) || data['type'] !== 'response.completed') return
    const response = data['response']
    if (!record(response) || !Array.isArray(response['output'])) return
    const items = response['output'].filter((item: unknown) => record(item) && item['type'] === 'image_generation_call') as Record<string, unknown>[]
    if (items.length === 0) return
    this.signal.throwIfAborted()
    const attachments = this.resolveAttachments?.()
    if (attachments === undefined) {
      throw new LlmError('Responses image output requires the durable attachment service', 'UNSUPPORTED_CONTENT')
    }
    if (items.length > attachments.imageLimits.maxImagesPerMessage) {
      throw new LlmError('Responses image output exceeds the attachment image count limit', 'UNSUPPORTED_CONTENT')
    }
    const images: EncodedImageAttachment[] = items.map((item) => {
      const result = item['result']
      if (item['status'] !== 'completed' || typeof result !== 'string' || result.length === 0) {
        throw new LlmError('Responses image generation completed without image data', 'UNSUPPORTED_CONTENT')
      }
      if (result.length > 4 * Math.ceil(attachments.imageLimits.maxImageBytes / 3)) {
        throw new LlmError('Responses image output exceeds the attachment image byte limit', 'UNSUPPORTED_CONTENT')
      }
      return { mediaType: mediaType(item['output_format']), data: result }
    })
    if (images.reduce((bytes, image) => bytes + image.data.length, 0)
      > 4 * Math.ceil(attachments.imageLimits.maxMessageImageBytes / 3) + 4 * images.length) {
      throw new LlmError('Responses image output exceeds the attachment batch byte limit', 'UNSUPPORTED_CONTENT')
    }
    const refs = await admitEncodedImages(attachments, images)
    this.signal.throwIfAborted()
    this.content = refs.map((ref) => {
      const path = this.resolveAccess?.(attachments, ref)?.readonlyPath
      if (path === undefined) {
        throw new LlmError('Responses image output requires a readable attachment path in the tool filesystem', 'UNSUPPORTED_CONTENT')
      }
      if (/[\u0000-\u001f\u007f]/.test(path)) {
        throw new LlmError('Responses image output requires an attachment path without control characters', 'UNSUPPORTED_CONTENT')
      }
      // Angle destinations admit spaces; encode Markdown delimiters and
      // filename characters the file renderer treats as queries or fragments.
      const destination = path.replace(/[%<>?#\\]/g, character => encodeURIComponent(character))
      return { type: 'text', text: `![Generated image](<${destination}>)` }
    })
  }

  /**
   * Append admitted links before the successful terminal event. Native text
   * and function-call indices remain unchanged; mixed responses retain pi-ai's
   * toolUse reason, so the normal agent tool loop continues.
   * @param events - this request's normalized pi-ai events.
   * @returns events with image links and index-aligned replay content.
   */
  async * events(events: AsyncIterable<AssistantMessageEvent>): AsyncGenerator<AssistantMessageEvent> {
    for await (const event of events) {
      if (event.type !== 'done' || this.content.length === 0) {
        yield event
        continue
      }
      const message = { ...event.message, content: [...event.message.content, ...this.content] }
      for (const [offset, block] of this.content.entries()) {
        const contentIndex = event.message.content.length + offset
        yield { type: 'text_start', contentIndex, partial: message }
        yield { type: 'text_delta', contentIndex, delta: block.text, partial: message }
        yield { type: 'text_end', contentIndex, content: block.text, partial: message }
      }
      yield { ...event, message }
    }
  }
}
