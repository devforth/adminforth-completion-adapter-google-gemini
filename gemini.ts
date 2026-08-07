import { GoogleGenAI } from "@google/genai";
import { logger } from "adminforth";
import pRetry from "p-retry";
import type { AdapterOptions } from "./types.js";

export type CompletionRequestInput = {
  content: string;
  maxTokens?: number;
  outputSchema?: any;
};

type GeminiUsageMetadata = {
  promptTokenCount?: number;
  cachedContentTokenCount?: number;
  candidatesTokenCount?: number;
  thoughtsTokenCount?: number;
  prompt_token_count?: number;
  cached_content_token_count?: number;
  candidates_token_count?: number;
  thoughts_token_count?: number;
};

export type UsedTokens = {
  input_uncached: number;
  input_cached: number;
  output: number;
};

export type CompletionResult = {
  content?: string;
  finishReason?: string;
  error?: string;
  used_tokens?: UsedTokens;
};

const JSON_CODE_FENCE_RE = /^```(?:json)?\s*([\s\S]*?)\s*```$/i;

function getResponseJsonSchema(outputSchema?: any) {
  return outputSchema?.schema ?? outputSchema;
}

function unwrapJsonCodeFence(content: string | undefined) {
  return content?.replace(JSON_CODE_FENCE_RE, "$1");
}

function extractUsedTokens(usageMetadata?: GeminiUsageMetadata): UsedTokens | undefined {
  if (!usageMetadata) return undefined;

  const input = usageMetadata.promptTokenCount ?? usageMetadata.prompt_token_count ?? 0;
  const inputCached =
    usageMetadata.cachedContentTokenCount ??
    usageMetadata.cached_content_token_count ??
    0;
  const output =
    usageMetadata.candidatesTokenCount ??
    usageMetadata.candidates_token_count ??
    0;
  const thoughts =
    usageMetadata.thoughtsTokenCount ??
    usageMetadata.thoughts_token_count ??
    0;

  return {
    input_uncached: Math.max(input - inputCached, 0),
    input_cached: inputCached,
    output: output + thoughts,
  };
}

export class GeminiService {
  constructor(private options: AdapterOptions) {}

  validate() {
    if (!this.options.geminiApiKey) {
      throw new Error("geminiApiKey is required");
    }
  }

  async measureTokensCount(content: string): Promise<number> {
    const ai = new GoogleGenAI({
      apiKey: this.options.geminiApiKey,
    });
    const countTokensResponse = await ai.models.countTokens({
      model: "gemini-2.0-flash",
      contents: content,
    });

    return countTokensResponse.totalTokens;
  }

  async complete(request: CompletionRequestInput): Promise<CompletionResult> {
    const {
      content,
      maxTokens: requestMaxTokens = 50,
      outputSchema: requestOutputSchema,
    } = request;
    const responseJsonSchema = getResponseJsonSchema(requestOutputSchema);
    const ai = new GoogleGenAI({
      apiKey: this.options.geminiApiKey,
    });

    const tryToGenerate = async () => {
      logger.debug("Making Google Gemini API call");
      try {
        const response = await ai.models.generateContent({
          model: this.options.model || "gemini-3-flash-preview",
          contents: [
            {
              role: "user",
              parts: [{ text: content }],
            },
          ],
          config: {
            maxOutputTokens: requestMaxTokens,
            ...this.options.extraRequestBodyParameters,
            ...(responseJsonSchema
              ? {
                  responseFormat: [
                    {
                      text: {
                        mimeType: "application/json",
                        schema: responseJsonSchema,
                      },
                    },
                  ],
                }
              : {}),
          },
        });
        logger.debug(`Google Gemini SUCCESSFUL API response: ${response}`);
        return {
          content: responseJsonSchema
            ? unwrapJsonCodeFence(response.text)
            : response.text,
          used_tokens: extractUsedTokens(response.usageMetadata),
        };
      } catch (error) {
        logger.error(`Error during Google Gemini API call: ${error}`);
        const errorMessage = error instanceof Error ? error.message : String(error);
        let googleErrorMessage = errorMessage;
        try {
          googleErrorMessage = JSON.parse(errorMessage).error.message;
        } catch {
        }
        throw new Error(`Error during Google Gemini API call: ${googleErrorMessage}`);
      }
    };

    return pRetry(tryToGenerate, {
      retries: 5,
      onFailedAttempt: ({ error, attemptNumber, retriesLeft, retriesConsumed }) => {
        logger.debug(`Attempt ${attemptNumber} failed. ${retriesLeft} retries left. ${retriesConsumed} retries consumed.`);
      },
    });
  }
}
