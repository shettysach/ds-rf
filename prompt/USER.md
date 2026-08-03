Choose one motion and direction for the robot's next action.

Return exactly one valid JSON object with exactly these two string fields:

```json
{"motion":"walk","direction":"forward"}
```

Allowed values:
- `motion`: `stand`, `walk`
- `direction`: `forward`, `backward`, `left`, `right`
- If `motion` is `stand`, set `direction` to `forward`.

Do not include any extra fields, Markdown, code fences, comments, explanation, or text outside the JSON object.
