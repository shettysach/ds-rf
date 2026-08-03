Choose one motion and direction for the robot's next action.

Return only one JSON object, with exactly these string fields:

```json
{"motion":"walk","direction":"left"}
```

Allowed motion values: `stand`, `walk`.

Allowed direction values: `forward`, `backward`, `left`, `right`.

For standing, use direction `forward`.

Do not return Markdown, punctuation outside the JSON object, or an explanation.
