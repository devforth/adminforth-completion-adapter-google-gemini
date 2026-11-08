/** Model used when `model` is not set. */
export const DEFAULT_MODEL = "gemini-3.8-flash";

export interface AdapterOptions {
  /**
   * Google API key. Go to https://aistudio.google.com/api-keys, go to Dashboard -> API keys -> Create new secret key
   * Set geminiApiKey: process.env.GEMINI_API_KEY to access it
   */
  geminiApiKey: string;

  /**
   * Model name. Go to https://ai.google.dev/gemini-api/docs/models, select model and copy name.
   * Default is `gemini-3.8-flash`. Use e.g. more expensive `gemini-4-argon` for more powerful model.
   */
  model?: string;


  /**
   * Additional request body parameters to include in the API request.
   */
  extraRequestBodyParameters?: Record<string, unknown>;
}
