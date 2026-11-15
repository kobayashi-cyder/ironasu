# ironasu Studio — オフラインで動くWeb UI

**ironasu Studio** は LM Studio風の操作感を持つ、追加ランタイム依存のないWeb UIです。
HTML / CSS / JavaScript はPythonパッケージに含まれています。CDN、Google Fonts、
クラウドAPI、インターネットアクセスは不要です。表示と既存のironasuゲートウェイは
完全にローカルで動かせます。

**重要:** UIはLLM実行エンジンではありません。モデル推論には別途起動したLocalAI、
llama.cpp server、LM StudioのOpenAI互換ローカルサーバーなどが必要です。
モデル、実行バイナリ、Pythonパッケージやwheelなどはオフライン環境に入る前に
ローカルPCへ配置してください。UI自体はバックエンド未起動でも開きます。

## 起動

ソースチェックアウトに、Python 3.10以上とビルド要件（setuptools>=68）
がインストール済みの場合:

```sh
python -m pip install --no-index --no-build-isolation -e .
# 別のターミナルでLocalAI / llama.cpp / LM Studio等のローカルOpenAI互換サーバーを起動
export IRONASU_BACKEND_URL=http://127.0.0.1:8080/v1
python -m ironasu
```

Webブラウザで **http://127.0.0.1:8765/ui** を開きます。既存のAPIは
`http://127.0.0.1:8765/v1/chat/completions` のままです。

Windows PowerShell:

```powershell
$env:IRONASU_BACKEND_URL = "http://127.0.0.1:8080/v1"
python -m ironasu
```

完全にネットワーク接続がないPCに移す場合、インターネット接続のある環境で
wheelとすべての依存を事前に準備するか、Python + setuptools>=68がすでにある
PCで `python -m pip install --no-index --no-build-isolation -e .` を利用します。
UI用の追加Python/Nodeライブラリは**不要**です。wheel配布時には
`ironasu/web/index.html`、`ui.css`、`ui.js` が同梱されます。

## 使える機能

- 会話の新規作成・切替・削除、JSONエクスポート、応答コピー。
- **自動 / 高速 / 高品質**ルート切替と、上流の `/v1/models` が返すモデルIDの直接指定。
- Temperature、Top P、Max tokens、System prompt の編集。
- 非ストリーミング生成と、対応バックエンドのSSEストリーミング生成・停止。
- `/metrics` に基づく現在の実行数、待機数、キャッシュヒット数、上流呼び出し回数、負荷状態。
- キーボード操作、狭い画面でのナビゲーション、サーバー接続状態表示。

## プライバシーと制約

UIはサーバーと**同じオリジン**の相対URLだけへアクセスし、外部URLを参照しません。
チャット履歴はデフォルトでメモリ内のみです。「このブラウザに履歴を保存」を
ONにした場合だけブラウザのlocalStorageに平文で保存されます。
エクスポートしたJSONにも会話全文が入ります。共有PCでは保存しないでください。

`IRONASU_SERVER_KEY` を設定した場合、ブラウザでBearer tokenを入力できます。
キーはブラウザ内で永続化しません。一般公開・インターネットへのポート開放は
推奨しません。ゲートウェイはデフォルトで127.0.0.1にバインドします。

LM Studioの公式UIや内部APIとの完全互換ではありません。モデルのダウンロード、
ロード/アンロード、GGUF変換、GPUメモリ制御は実装していません。モデルの一覧は
ローカルバックエンドが提供するときのみ取得され、見つからなくても自動ルーティング
によるチャットは利用できます。

`stream=true` は**SSE対応バックエンドだけ**で使えます。レガシーの
コードコンテキスト選択 `x_ironasu_files` とストリーミングは併用できません。
ストリーミング応答はバイトを透過転送するため `_ironasu` メタデータが付かず、
キャッシュ/シングルフライトの対象外です。推論の負荷制限は共通です。

APIからの直接呼び出しでは以下を利用できます。

```sh
curl http://127.0.0.1:8765/v1/models
curl http://127.0.0.1:8765/ui/status
curl http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ironasu-fast","messages":[{"role":"user","content":"こんにちは"}]}'
```

モデルの直接指定を有効にする場合は、ゲートウェイ固有の
`x_ironasu_model` に上流のモデルIDを渡します。
