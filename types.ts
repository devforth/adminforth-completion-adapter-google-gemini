export interface AdapterOptions {
  /**
   * Google API key. Go to https://aistudio.google.com/api-keys, go to Dashboard -> API keys -> Create new secret key
   * Set geminiApiKey: process.env.GEMINI_API_KEY to access it
   */
  geminiApiKey: string;

  /**
   * Model name. Go to https://ai.google.dev/gemini-api/docs/models, select model and copy name.
   * Default is `gemini-3-flash-preview`. Use e.g. more expensive `gemini-3-pro-preview` for more powerful model.
   */
  model?: string;


  /**
   * Additional request body parameters to include in the API request.
   */
  extraRequestBodyParameters?: Record<string, unknown>;

  /**
   * Expert settings
   */
  expert?: {
    /**
     * Temperature (0-1). Lower is more deterministic, higher is more unpredicted creative. Default is 0.7.
     */
    temperature?: number;
  };
}
