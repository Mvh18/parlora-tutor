/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the Parlora Tutor server; defaults to the local Compose stack. */
  readonly VITE_PARLORA_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
