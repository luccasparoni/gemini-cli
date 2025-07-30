/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  BaseTool,
  ToolResult,
  ToolCallConfirmationDetails,
  ToolConfirmationOutcome,
  ToolMcpConfirmationDetails,
  Icon,
} from './tools.js';
import {
  CallableTool,
  Part,
  FunctionCall,
  FunctionDeclaration,
  Type,
} from '@google/genai';

type ToolParams = Record<string, unknown>;

/**
 * MCP content types that contain media data
 */
const MCP_MEDIA_TYPES = ['image', 'pdf', 'audio', 'video'] as const;
type McpMediaType = typeof MCP_MEDIA_TYPES[number];

export class DiscoveredMCPTool extends BaseTool<ToolParams, ToolResult> {
  private static readonly allowlist: Set<string> = new Set();

  constructor(
    private readonly mcpTool: CallableTool,
    readonly serverName: string,
    readonly serverToolName: string,
    description: string,
    readonly parameterSchemaJson: unknown,
    readonly timeout?: number,
    readonly trust?: boolean,
    nameOverride?: string,
  ) {
    super(
      nameOverride ?? generateValidName(serverToolName),
      `${serverToolName} (${serverName} MCP Server)`,
      description,
      Icon.Hammer,
      { type: Type.OBJECT }, // this is a dummy Schema for MCP, will be not be used to construct the FunctionDeclaration
      true, // isOutputMarkdown
      false, // canUpdateOutput
    );
  }

  asFullyQualifiedTool(): DiscoveredMCPTool {
    return new DiscoveredMCPTool(
      this.mcpTool,
      this.serverName,
      this.serverToolName,
      this.description,
      this.parameterSchemaJson,
      this.timeout,
      this.trust,
      `${this.serverName}__${this.serverToolName}`,
    );
  }

  /**
   * Overrides the base schema to use parametersJsonSchema when building
   * FunctionDeclaration
   */
  override get schema(): FunctionDeclaration {
    return {
      name: this.name,
      description: this.description,
      parametersJsonSchema: this.parameterSchemaJson,
    };
  }

  async shouldConfirmExecute(
    _params: ToolParams,
    _abortSignal: AbortSignal,
  ): Promise<ToolCallConfirmationDetails | false> {
    const serverAllowListKey = this.serverName;
    const toolAllowListKey = `${this.serverName}.${this.serverToolName}`;

    if (this.trust) {
      return false; // server is trusted, no confirmation needed
    }

    if (
      DiscoveredMCPTool.allowlist.has(serverAllowListKey) ||
      DiscoveredMCPTool.allowlist.has(toolAllowListKey)
    ) {
      return false; // server and/or tool already allowlisted
    }

    const confirmationDetails: ToolMcpConfirmationDetails = {
      type: 'mcp',
      title: 'Confirm MCP Tool Execution',
      serverName: this.serverName,
      toolName: this.serverToolName, // Display original tool name in confirmation
      toolDisplayName: this.name, // Display global registry name exposed to model and user
      onConfirm: async (outcome: ToolConfirmationOutcome) => {
        if (outcome === ToolConfirmationOutcome.ProceedAlwaysServer) {
          DiscoveredMCPTool.allowlist.add(serverAllowListKey);
        } else if (outcome === ToolConfirmationOutcome.ProceedAlwaysTool) {
          DiscoveredMCPTool.allowlist.add(toolAllowListKey);
        }
      },
    };
    return confirmationDetails;
  }

  async execute(params: ToolParams): Promise<ToolResult> {
    const functionCalls: FunctionCall[] = [
      {
        name: this.serverToolName,
        args: params,
      },
    ];

    const responseParts: Part[] = await this.mcpTool.callTool(functionCalls);
    
    // Transform MCP media content to Gemini format
    const transformedParts = this.transformMcpResponse(responseParts);

    return {
      llmContent: transformedParts,
      returnDisplay: getStringifiedResultForDisplay(responseParts),
    };
  }

  /**
   * Transforms MCP tool responses to properly handle media content.
   * Extracts media content as separate inlineData Parts that Gemini can see.
   * Preserves text content as separate text Parts.
   * Handles resource content from MCP spec.
   * Maintains backward compatibility for text-only responses.
   */
  private transformMcpResponse(responseParts: Part[]): Part[] {
    const result: Part[] = [];
    let hasTransformedContent = false;
    let currentPartHasMedia = false;
    let currentPartItems: Part[] = [];
    
    // Handle multiple parts (though MCP typically returns single part)
    for (const part of responseParts) {
      if (!part.functionResponse) {
        // Not a function response, keep as-is
        result.push(part);
        continue;
      }
      
      const functionResponse = part.functionResponse;
      const content = functionResponse.response?.content;
      
      if (!content || !Array.isArray(content)) {
        // No content to transform, keep original
        result.push(part);
        continue;
      }
      
      // Check if this is an error response
      if (functionResponse.response?.isError) {
        const errorText = content.find(item => this.isTextContent(item))?.text || 'Unknown error';
        result.push({ text: `Error from ${this.serverToolName}: ${errorText}` });
        hasTransformedContent = true;
        continue;
      }
      
      // Reset for this part
      currentPartHasMedia = false;
      currentPartItems = [];
      
      // Transform each content item into a separate Part
      for (const item of content) {
        if (this.isTextContent(item)) {
          currentPartItems.push({ text: item.text });
          hasTransformedContent = true;
        } else if (this.isMediaContent(item)) {
          // Validate media data before using
          if (!item.data || !item.mimeType || 
              typeof item.data !== 'string' || 
              typeof item.mimeType !== 'string') {
            currentPartItems.push({ text: '[Invalid media: missing data or mimeType]' });
            hasTransformedContent = true;
            continue;
          }
          
          currentPartHasMedia = true;
          hasTransformedContent = true;
          currentPartItems.push({
            inlineData: {
              data: item.data,
              mimeType: item.mimeType,
            },
          });
        } else if (this.isResourceContent(item)) {
          // Handle embedded resource content
          const resource = item.resource;
          if (resource.blob && resource.mimeType) {
            // Resource has binary data
            currentPartHasMedia = true;
            hasTransformedContent = true;
            currentPartItems.push({
              inlineData: {
                data: resource.blob,
                mimeType: resource.mimeType,
              },
            });
          } else if (resource.text) {
            // Resource has text data
            currentPartItems.push({ text: `[Resource: ${resource.uri}]\n${resource.text}` });
            hasTransformedContent = true;
          } else {
            // Resource only has URI
            currentPartItems.push({ text: `[Resource: ${resource.uri}]` });
            hasTransformedContent = true;
          }
        } else if (item && typeof item === 'object' && 'type' in item) {
          const itemType = (item as Record<string, unknown>).type;
          // Check if this is an invalid media type (has media type but missing/invalid data)
          if (typeof itemType === 'string' && MCP_MEDIA_TYPES.includes(itemType as McpMediaType)) {
            currentPartItems.push({ text: '[Invalid media: missing data or mimeType]' });
            hasTransformedContent = true;
          } else {
            // Unknown content type - include as text with warning
            currentPartItems.push({ text: `[Unknown content type: ${itemType}]` });
            hasTransformedContent = true;
          }
        }
      }
      
      // If this part has media, prepend context
      if (currentPartHasMedia && currentPartItems.length > 0) {
        result.push({ 
          text: `[Response from ${this.serverToolName} (${this.serverName} MCP Server)]` 
        });
      }
      
      // Add all items from this part
      result.push(...currentPartItems);
    }
    
    // If we transformed content, return the new Parts
    // Otherwise, keep the original parts for backward compatibility
    return hasTransformedContent ? result : responseParts;
  }

  /**
   * Type guard for MCP text content
   */
  private isTextContent(item: unknown): item is { type: string; text: string } {
    return (
      typeof item === 'object' &&
      item !== null &&
      'type' in item &&
      'text' in item &&
      (item as Record<string, unknown>).type === 'text' &&
      typeof (item as Record<string, unknown>).text === 'string'
    );
  }

  /**
   * Type guard for MCP resource content
   */
  private isResourceContent(item: unknown): item is { type: string; resource: { uri: string; mimeType?: string; text?: string; blob?: string } } {
    return (
      typeof item === 'object' &&
      item !== null &&
      'type' in item &&
      'resource' in item &&
      (item as Record<string, unknown>).type === 'resource' &&
      typeof (item as Record<string, unknown>).resource === 'object' &&
      typeof ((item as Record<string, unknown>).resource as Record<string, unknown>).uri === 'string'
    );
  }

  /**
   * Type guard for MCP media content (image, audio, video, pdf per MCP spec)
   */
  private isMediaContent(item: unknown): item is { type: string; data: string; mimeType: string } {
    if (typeof item !== 'object' || item === null) {
      return false;
    }
    
    const obj = item as Record<string, unknown>;
    
    // Must have all required fields
    if (!('type' in obj) || !('data' in obj) || !('mimeType' in obj)) {
      return false;
    }
    
    // Check if type is a valid media type
    if (typeof obj.type !== 'string' || !MCP_MEDIA_TYPES.includes(obj.type as McpMediaType)) {
      return false;
    }
    
    // For type guard to work properly, data and mimeType must be non-null
    // If they are null/undefined, it's not valid media content even if type is correct
    return obj.data !== null && obj.data !== undefined && 
           obj.mimeType !== null && obj.mimeType !== undefined;
  }
}

/**
 * Processes an array of `Part` objects, primarily from a tool's execution result,
 * to generate a user-friendly string representation, typically for display in a CLI.
 *
 * The `result` array can contain various types of `Part` objects:
 * 1. `FunctionResponse` parts:
 *    - If the `response.content` of a `FunctionResponse` is an array consisting solely
 *      of `TextPart` objects, their text content is concatenated into a single string.
 *      This is to present simple textual outputs directly.
 *    - If `response.content` is an array but contains other types of `Part` objects (or a mix),
 *      the `content` array itself is preserved. This handles structured data like JSON objects or arrays
 *      returned by a tool.
 *    - If `response.content` is not an array or is missing, the entire `functionResponse`
 *      object is preserved.
 * 2. Other `Part` types (e.g., `TextPart` directly in the `result` array):
 *    - These are preserved as is.
 *
 * All processed parts are then collected into an array, which is JSON.stringify-ed
 * with indentation and wrapped in a markdown JSON code block.
 */
function getStringifiedResultForDisplay(result: Part[]) {
  if (!result || result.length === 0) {
    return '```json\n[]\n```';
  }

  const processFunctionResponse = (part: Part) => {
    if (part.functionResponse) {
      const responseContent = part.functionResponse.response?.content;
      if (responseContent && Array.isArray(responseContent)) {
        // Process each content item
        const processedContent = responseContent.map((item: unknown) => {
          if (!item || typeof item !== 'object') {
            return item;
          }
          
          const obj = item as Record<string, unknown>;
          
          // Handle MCP text content
          if (obj.type === 'text' && obj.text) {
            return obj.text;
          }
          // Handle MCP media content with user-friendly display
          else if (obj.type === 'image' && obj.data && obj.mimeType) {
            return `[Image: ${obj.mimeType}]`;
          }
          else if (obj.type === 'audio' && obj.data && obj.mimeType) {
            return `[Audio: ${obj.mimeType}]`;
          }
          else if (obj.type === 'video' && obj.data && obj.mimeType) {
            return `[Video: ${obj.mimeType}]`;
          }
          else if (obj.type === 'pdf' && obj.data && obj.mimeType) {
            return `[PDF document]`;
          }
          // Handle text-only Parts (backward compatibility)
          else if (obj.text !== undefined) {
            return obj.text;
          }
          // Unknown content type
          return item;
        });
        
        // If all items are strings, join them
        if (processedContent.every((item: unknown) => typeof item === 'string')) {
          return processedContent.join('\n');
        }
        
        // Otherwise return the processed array
        return processedContent;
      }

      // If no content, or not an array, or not a functionResponse, stringify the whole functionResponse part for inspection
      return part.functionResponse;
    }
    return part; // Fallback for unexpected structure or non-FunctionResponsePart
  };

  const processedResults =
    result.length === 1
      ? processFunctionResponse(result[0])
      : result.map(processFunctionResponse);
  if (typeof processedResults === 'string') {
    return processedResults;
  }

  return '```json\n' + JSON.stringify(processedResults, null, 2) + '\n```';
}

/** Visible for testing */
export function generateValidName(name: string) {
  // Replace invalid characters (based on 400 error message from Gemini API) with underscores
  let validToolname = name.replace(/[^a-zA-Z0-9_.-]/g, '_');

  // If longer than 63 characters, replace middle with '___'
  // (Gemini API says max length 64, but actual limit seems to be 63)
  if (validToolname.length > 63) {
    validToolname =
      validToolname.slice(0, 28) + '___' + validToolname.slice(-32);
  }
  return validToolname;
}
