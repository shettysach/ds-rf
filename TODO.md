[ ] ONNX - try TensorrtExecutionProvider vs CUDAExecutionProvider (current) 

[ ] Bound VLM history without defeating llama-server's KV-prefix cache
    - Keep history append-only within an epoch; do not use a turn-by-turn sliding
      window, because evicting the oldest turn changes the prompt immediately after
      the system message and forces retained image turns to be prefetched again.
    - Explicitly send `cache_prompt: true` even though the server currently enables
      it by default.
    - Log `timings.cache_n`, `timings.prompt_n`, `timings.prompt_ms`,
      `timings.predicted_n`, and `timings.predicted_ms` from llama-server responses.
    - Compare append-only, stateless, and sliding-window histories using the same
      observations before selecting a policy.
    - Roll over to a new history epoch once a context budget is reached. If long-term
      state becomes necessary, create a stable text summary only at rollover rather
      than rewriting it every turn.
    - Avoid retaining and base64-encoding every old JPEG indefinitely; epoch rollover
      must bound client memory, JSON serialization, and HTTP payload size as well as
      model context usage.
    - The normal server invocation already uses one slot (`--parallel 1`, equivalent
      to `-np 1`) and prompt caching (`--cache-prompt`).
