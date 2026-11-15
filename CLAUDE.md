# CLAUDE.md

## Kaggle 関連のスライド作成

- Kaggle のコンペ・分析結果などのスライドを作るときは、必ず anthropics/skills の **pptx スキル**（`document-skills@anthropic-agent-skills` の `pptx`）を使って `.pptx` ファイルとして作成する。
  - クラウドセッションで `anthropic-skills:pptx` として提供されている場合はそれを使ってよい（同じスキル）。
- 他のスライド形式（Artifact のスライドデッキ型など）は、ユーザーが明示的に指定したときだけ使う。
