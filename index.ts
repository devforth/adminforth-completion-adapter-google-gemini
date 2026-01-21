import type { AdapterOptions } from "./types.js";
import type { CompletionAdapter } from "adminforth";
import { GoogleGenAI } from "@google/genai";


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

    try {
      const response = await ai.models.generateContent({
        model: this.options.model || "gemini-3-flash-preview",
        contents: content,
        config: {
          temperature: this.options.expert?.temperature ?? 0.7,
          maxOutputTokens: maxTokens,
          stopSequences: stop,
          ...this.options.extraRequestBodyParameters,
        },
      });
      return {
        content: response.text,
      };
    } catch (error) {
      return { error: (error as Error).message };
    }
  };
}
