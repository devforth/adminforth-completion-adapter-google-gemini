import type { AdapterOptions } from "./types.js";
import type { CompletionAdapter } from "adminforth";
import { GoogleGenAI } from "@google/genai";
import pRetry from 'p-retry';
import { logger } from "adminforth";

export default class CompletionAdapterGoogleGemini
  implements CompletionAdapter
{
  options: AdapterOptions;

  constructor(options: AdapterOptions) {
    this.options = options;
  }
  
  validate() {
    if (!this.options.geminiApiKey) {
      throw new Error("geminiApiKey is required");
    }
  }

  complete = async (content: string, stop = ["."], maxTokens = 50): Promise<{
    content?: string;
    finishReason?: string;
    error?: string;
  }> => {

    const ai = new GoogleGenAI({
      apiKey: this.options.geminiApiKey,
    });

    const tryToGenerate = async () => {
      logger.debug("Making Google Gemini API call");
      try {
        const response = await ai.models.generateContent({
          model: this.options.model || "gemini-3-flash-preview",
          contents: content,
          config: {
            maxOutputTokens: maxTokens,
            ...this.options.extraRequestBodyParameters,
          },
        });
        logger.debug(`Google Gemini SUCCESSFUL API response: ${response}`);
        return {
          content: response.text,
        };
      } catch (error) {
        logger.error(`Error during Google Gemini API call: ${error}`);
        throw new Error(`Error during Google Gemini API call: ${JSON.parse(error.message).error.message}`);
      }
    }
    const result = await pRetry(tryToGenerate, {
      retries: 5,
      onFailedAttempt: ({error, attemptNumber, retriesLeft, retriesConsumed}) => {
        logger.debug(`Attempt ${attemptNumber} failed. ${retriesLeft} retries left. ${retriesConsumed} retries consumed.`);
      },
    })
    return result;
  };
}
