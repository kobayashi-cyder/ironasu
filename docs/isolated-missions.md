# ironasu v1.9 — Git worktree隔離と独立検証

**実験的・明示的なオプトイン機能**です。従来の Mission Controller はそのまま使用できます。
Qt6「ミッション」タブの「隔離Git worktreeで並列編集」をオンにするか、CLI の
`--isolated-worktrees` を指定した場合のみ、元のブランチを書き換えずに最大16作業ツリーへ編集を分散します。

## 動作

1. Gitリポジトリのルートを確認し、作業ツリーに未保存の変更がないことを検証します（`.ironasu/`内の任務データは対象外）。
2. 任務開始時の Git HEAD を記録。検証用のbaseline worktreeを作成し、**変更前**の指定テスト・期待成果物の結果をSQLiteへ保存します。
3. タスクDAGに従い独立 worktreeを作成。依存タスクのコミットをGit mergeで受け取り、各ワーカーに独立した編集対象を渡します。モデル同時推論枠は別途制限します。
4. 各ワーカーの変更を、ファイル数・サイズ・保護パス・シンボリックリンク・削除・差分エラーを検査してから **その worktree のみ**にコミットします。
5. 最終タスクのコミットを別の `candidate` worktreeへGit merge。競合時は停止し、自動的にコンフリクトを解消したり元ブランチを編集したりしません。
6. 候補の差分を元HEADと比較して検査し、**変更後**の指定テストと成果物の存在/SHA-256を検証します。検証失敗時は明示的な権限のもとで候補worktreeだけを上限回数修復します。
7. 成功しても状態は `verified_isolated`（**検証済み・未統合**）。元ブランチへの反映は別アクションの明示承認が必要です。

## CLI例

```powershell
cd C:\Users\hazim\ironasu
.\.venv\Scripts\python.exe -m pip install -e ".[desktop]"

ironasu-mission "このリポジトリの不具合を修正しテストしてください" `
  --workspace . `
  --model qwen3.5-4b-weight-space-programming `
  --workers 4 --model-slots 2 `
  --allow-writes --allow-tests --isolated-worktrees `
  --verify unittest --verify git_diff_check `
  --max-repairs 2
```

Gitの履歴や他のブランチを勝手に公開しません。候補が検証済みになった後、差分とレポートをレビューし、変更の反映を明示的に承認するには次を実行します。

```powershell
ironasu-mission --workspace . --mission-id YOUR_MISSION_ID `
  --approve-integration
```

反映には、元のHEADが任務開始時から不変で、作業ツリーがクリーンで、候補が検証済みであることが必要です。Gitの **fast-forwardのみ** 許可します。競合や中断があれば人間による調査を要求します。処理開始前に統合操作をSQLiteへ記録するため、クラッシュ後に不確実な統合を自動再試行しません。

停止・再開：

```powershell
ironasu-mission --workspace . --mission-id YOUR_MISSION_ID --resume `
  --isolated-worktrees --allow-writes --allow-tests `
  --model qwen3.5-4b-weight-space-programming
```

再開時は同じモデルURL・モデル名・作業フォルダーを指定します。成功済みタスクのGitコミットとモデル履歴を再利用し、操作結果が不確かな書き込みはハーネスのSQLiteジャーナルで保留します。通常モードで隔離任務を再開することは拒否します。

## 何を検証するか

ベースラインと候補の両方に、指定された `unittest` / `pytest` / `compileall` / `git_diff_check` を独立して実行します。Gitコミット済みの差分は別途 `git diff --check BASE CANDIDATE` でも検査し、実際に変更されたファイルとパッチのSHA-256を記録します。

**注意：** `compileall` は構文検査であり機能の保証ではありません。すべてのテストを通過してもテストされていない要求の達成は保証できません。正確な実行速度やGrok Botとの優劣も別途ベンチマークが必要です。

## 安全上の限界

- Git worktree は**ファイルの作業領域を隔離する仕組み**であり、OS/コンテナの安全なサンドボックスではありません。検証テストやGitフィルターが他のファイル・ネットワーク等にアクセスする可能性があります。信頼できるリポジトリにだけ実行してください。
- 初期状態では隔離並列編集はオフ。使用するにはファイル変更とテスト実行の**両方**の許可が必要です。テスト実行はそのリポジトリのコードを実際に動かします。
- 任意シェル実行、Git push、公開、メール送信等をモデルへツール提供しません。Gitの内部コミット/merge処理だけが定義されたコマンド引数で行われます。Git hookは隔離操作中無効化しますが、信頼できないGit設定・外部プログラムをOSレベルで封じるものではありません。
- タスクごとの自動変更は1回100ファイル・合計10MB以下に制限し、削除・シンボリックリンク・プライベートパスの変更を拒否します。
- 作業結果と候補worktreeは `<repository>/.ironasu/worktrees/<missionID>` に残ります。ディスク容量が増えるため、確認後はGit worktreeコマンドを用いて人間が整理してください。勝手に削除しません。
- 複数依存の変更を自動的に一つにできなければ、**merge_conflict / needs_review** として報告します。誤った自動解決は実施しません。
- CPU/GPUの16倍並列化を意味しません。1台のRadeon環境ではモデル同時呼び出し2枠からの計測を推奨します。

[Mission Controllerガイド](mission.md)も参照してください。
