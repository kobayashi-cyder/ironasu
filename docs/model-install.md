# ironasu Studio: モデルのインストール（v1.4）

**v1.4以降はLM Studio/Ollamaを使わず、公開GGUFをironasuに直接保存できます。**
モデル管理画面の「ironasuへ直接保存」を選択してください。詳しくは
[ネイティブGGUFインストール](native-install.md)を参照してください。

以下はv1.3から継続提供するLM Studio/Ollama経由の取得方法です。

ironasu Studioの右上に追加された **「＋ モデルを追加」** から、ローカルモデルサーバー自身に
モデルを取得させます。**ironasuが巨大な重みファイルを再ダウンロードしたり、
モデル管理用の独自ディレクトリにコピーしたりするわけではありません。**
そのため、LM Studio / Ollamaの公式な保存先と再利用・検証・管理方式を維持できます。

インストールにはモデルサーバーの起動とインターネット接続・十分なディスク容量が必要です。
**ダウンロード済みモデルとランタイムがあれば、推論自体はオフラインで利用できます。**
モデル形式、必要メモリ、GPU互換性、ライセンスは各モデル・バックエンドに依存します。

## LM Studio（公式v1 REST API）

1. LM StudioのDeveloper画面からローカルサーバーを起動（既定 `127.0.0.1:1234`）。
2. ironasu Studioの「＋ モデルを追加」を開き、**LM Studio**を選択。
3. LM StudioモデルカタログのID（例：`ibm/granite-4-micro`）を入力。
   あるいは正確なHugging FaceのモデルリポジトリURL
   （例：`https://huggingface.co/owner/model-GGUF`）を入力します。
   GGUF等の量子化を選ぶ場合、Hugging Face URLに対してだけ `Q4_K_M` 等を入力できます。
4. ダウンロード開始を押して進捗を確認します。完了したら
   「インストール済みモデル」の **ロード** を押します。
5. 右上のモデル一覧が更新されたら、モデルを選択してチャットします。

使用する公式API：
- `POST /api/v1/models/download`
- `GET /api/v1/models/download/status/{job_id}`
- `GET /api/v1/models`
- `POST /api/v1/models/load`

公式参考資料：
- https://lmstudio.ai/docs/developer/rest/download
- https://lmstudio.ai/docs/developer/rest/download-status
- https://lmstudio.ai/docs/developer/rest/load

**LM Studioのv1 APIを備えたバージョンが必要です（LM Studio 0.4以降）。**
古いバージョンは直接インストールAPIに対応していないことがあります。認証を設定した場合、
ironasuの `IRONASU_BACKEND_KEY` に同じトークンを設定してください。
インストール用APIは既定の `127.0.0.1:1234` に接続します。

## Ollama（公式Pull API）

1. Ollamaを起動し、ローカルAPI `127.0.0.1:11434` を利用できるようにします。
2. モデル管理画面から **Ollama** を選び、Ollamaライブラリのモデル名
   （例：`qwen3:4b`）を入力して開始します。
3. 画面で取得の進捗（現在のレイヤーのバイト数）が更新されます。
4. `success` が返ると完了です。Ollamaは初回推論時にモデルをロードするため、
   LM Studioのような別のロード操作は必要ありません。

使用する公式API：`POST http://127.0.0.1:11434/api/pull`（NDJSONストリーム）、
`GET http://127.0.0.1:11434/api/tags`。

参考：https://docs.ollama.com/api/pull

## 注意点

- 本機能は**LM StudioとOllama専用**です。llama.cpp、LocalAI、vLLM、Janは
  推論API接続には対応しますが、共通のモデルダウンロードAPIはありません。
  それぞれの管理手段でダウンロードしてください。
- モデルIDを入力して開始する方式です。オンラインカタログの横断検索や
  ライセンス同意・モデル推奨スコアは未実装です。画面に表示するモデルID例は
  適合性や安全性の保証ではありません。
- ダウンロード処理はローカルのLM Studio/Ollamaに委譲されます。画面を閉じても
  サーバー側で続く場合があります。履歴はironasuプロセスのメモリにのみ保存され、
  再起動すると消えます。Ollamaはバックエンド側で中断した取得を再開できます。
- 2つまで同時ダウンロードを開始できます。上限超過時は429が返ります。
- LM Studioでダウンロードが完了しても、GPUメモリへの**ロードは自動で行いません**。
  ロードはユーザーが「ロード」を押した場合のみです。
- HTTP APIはironasuが管理する同一オリジン内のUIから利用します。
  ダウンロード先はコード上で指定された **127.0.0.1:1234 と 127.0.0.1:11434に固定**され、
  ユーザーが任意のネットワークURLをAPI経由で指定することはできません。
- UI自体はダウンロード画面を含めて完全にオフラインで起動できます。
  ただし新規モデルをインターネット経由で取得するには接続が必要です。
- ダウンロードジョブのエラーは進捗画面に表示されます。詳細な保存先・ディスク不足・
  アクセス権の問題はLM Studio/Ollama側のログを確認してください。

モデル管理画面は `http://127.0.0.1:8765/ui` から開けます。
別ポートで起動した場合はそのポートを利用してください。
