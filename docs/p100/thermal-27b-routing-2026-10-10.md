# Unsloth 27Bの温度切替とDSHのモデル選択

## 適用した設定

Unslothの `unsloth/Qwen3.8-27B-GGUF` は、ダウンロード済みの `UD-Q4_K_XL` を明示的に読み込みます。共通窓口は `http://127.0.0.1:8877/v1` です。

| 状態 | GPU設定 | GPU layers | コンテキスト | KV | batch / ubatch |
| --- | --- | --- | --- | --- | --- |
| 通常 | 物理GPU `[1, 0]`、P100先頭、配分3:2 | 1024 = 全層 | 65,536 | F16 | 512 / 512 |
| 高温時 | 物理GPU `[0]`、RTX 4070とCPU | 20 | 65,536 | F16 | 512 / 512 |

visionを無効化せず、コンテキストやKVの精度を下げることで退避させる変更はしていません。20層は、このPCのRTX容量、GGUFとvision projectorのサイズ、GGUFのblock countから作成した設定です。モデル名による専用分岐はありません。

温度制限フラグ、温度制限カウンターの増加、または78C以上で退避を予約します。68C以下が120秒続くと復帰を予約します。実行中の生成をキャンセルせず、読み込み中・学習中・複数モデル稼働中は待機します。読み込み直前にもモデルを再確認し、`force_cancel_active=false` を指定します。APIが409を返す競合も次の区切りまで待機します。

共通サービスはUnslothの監視を管理します。Unslothを通常のショートカットから開いた場合も、共通サービスが稼働していれば監視できます。従来のP100ショートカットの監視とはファイルロックで重複を防ぎます。監視用stopファイルによる明示的な停止は、自動で取り消しません。

## 27B選択後に9Bが動いた原因

インストール済みDSHの `dsh-compaction-basic/lib/index.js` の `summarizeWithLlm` は、要約モデルの指定がないと、`agent.session.requestHeader()?.config` に記録された前回のモデルを使います。次回用に27Bを選んでも、前回9Bの会話を圧縮する段階では9Bが呼ばれます。会話の保存状態にも、前回使用した9Bと次回選択した27Bが併存していました。

さらに旧 `lmstudio` providerには9Bだけ、`lmstudio-native` providerには27Bだけが登録されていました。通常のLM Studio設定へ切り替えると、9Bを選ぶ構成になっていました。

両providerに27Bと9Bを登録し、どちらも共通APIへ接続します。LM Studio用の会話要約は27Bへ明示的に固定しました。9Bをチャットに選んだ場合でも、このLM Studio用の要約は27Bです。Unslothの要約には対応するUnslothモデルを指定し、意図しないアプリへの要約移動を防ぎます。27Bの出力上限49,152と、既存の会話圧縮設定は維持します。

共通サービスは、SDKのモデル名とnativeプロセスの `--model` に指定されたGGUFファイルを照合します。不一致のモデルを代用せず、明示的に拒否します。実リクエストの検証でも、DSHが送信したmodelとnative応答の実ファイル名を確認します。

## その他の修正

- Unslothの監視対象が9BのUD量子化だけだったため、保存済み27B・9BのモデルID、量子化指定、ファイル指定を登録しました。
- RTXへの退避で全層GPUが指定されていたため、容量に応じたCPU併用を追加しました。競合するGPU layersやfitの追加引数も取り除きます。
- GPU IDが同じ場合でも、GPU layersが違えば再読み込みします。
- LM StudioのSDK上のidleだけで判断せず、nativeの `/slots` も確認します。共通API経由の生成はSDKのキュー表示に出ないためです。
- 明示的なアプリ切替リクエストでは、待機中のもう一方のアプリを安全にアンロードします。監視だけの動作ではアプリ間のアンロードをしません。
- 古い未使用会話に表示名をモデルIDとして保存した設定が1件あったため、DSHを終了した状態で正しいIDの選択イベントを追記しました。元のログとキャッシュをバックアップし、インストール済みDSHのreaderで再読込して確認しました。過去に使用したモデルの記録は書き換えません。

設定・認証情報・会話ログはユーザーディレクトリに保存し、Gitへ含めません。

## 再適用と検証

UnslothとLM Studioの生成が終了していることを確認して、共通サービスを停止します。Unsloth単独の監視を起動している場合は、その監視も停止して元のプロファイルを復元してください。

```powershell
scripts/p100/start-desktop-service.ps1 -Stop
```

DSH設定の更新は、DSHのElectronをNodeモードで使って `scripts/p100/configure_harness_gateway.cjs` を実行します。`--dry-run` は変更予定だけを出力します。通常のnode.exeからはapp.asar内のDSH依存を読み込めないため、この実行方法が必要です。

```powershell
& "$env:USERPROFILE/.unsloth/studio/unsloth_studio/Scripts/python.exe" -X utf8 scripts/p100/configure_unsloth_thermal.py --dry-run
& "$env:USERPROFILE/.unsloth/studio/unsloth_studio/Scripts/python.exe" -X utf8 scripts/p100/configure_unsloth_thermal.py
& "$env:USERPROFILE/.unsloth/studio/unsloth_studio/Scripts/python.exe" -X utf8 scripts/p100/install_desktop_service.py --update
scripts/p100/start-desktop-service.ps1
```

`configure_unsloth_thermal.py` はインストール済みUnsloth環境と、このcheckoutのgguf-pyを使います。新しい依存は追加しません。新しいモデルでは、通常のP100用設定と使用する量子化を保存してから再実行してください。複数の候補がある場合は推測して選びません。

`scripts/p100/audit_harness_models.cjs` はprovider、要約モデル、各会話の保存された選択を確認します。認証情報や会話本文を出力しません。無効な未使用の選択だけを修正する場合は、DSHを終了してから `repair_harness_selection.cjs --apply` を実行します。稼働中のDSHがあれば拒否します。

`validate_harness_gateway.cjs --provider lmstudio` はDSHの実LlmRuntimeを使い、モデル名・共通URL・native実ファイル・応答を確認します。`validate_unsloth_thermal.py` は、退避中の実モデルについて実温度による復帰を最大15分監視し、復帰後に共通APIから短い生成を確認します。冷却条件を満たさない場合は復帰を成功扱いにしません。

## 検証記録

2026-10-10の自動テストは147件中145件成功、2件は既存の条件付きskipでした。Ruff、Nodeの構文検査、git diffの空白検査も確認しています。

DSH実ランタイムから27Bへ短い生成を行い、送信モデル `qwen/qwen3.8-27b` と応答の `Qwen3.8-27B-Q4_K_M.gguf` が一致して正常stopしたことを確認しました。

P100が81Cで温度制限を検出した後、LM StudioはRTX 4070へ自動退避しました。その高温状態でUnsloth 27Bも共通APIから読み込み、HTTP 200・回答42・正常stopを確認しました。実状態は `UD-Q4_K_XL`、RTX `[0]`、要求GPU20層・実offload20/66、65,536、F16、vision有効、512/512でした。共有メモリの注入も維持されていました。

実温度が68C以下で安定した後、14:10:26 UTCにUnsloth 27Bの `switched-primary` を記録しました。復帰後はP100先頭の `[1, 0]`、要求1024層・実offload66/66、65,536、F16、vision有効でした。共通APIから回答42が正常stopし、冷却信号を模擬せず退避から復帰まで確認できました。

通常の `lmstudio` providerからもDSH実LlmRuntimeで27Bへの生成を行いました。送信モデルと27Bの実GGUFが一致し、回答42・正常stopでした。Unslothの待機中モデルを自動アンロードしてからLM Studioへ移る経路も、この実リクエストで確認しました。背景要約は検証後に有効へ戻し、共有メモリ・温度監視を稼働状態にしています。

保存された4会話の次回モデル選択は、監査時点ですべて有効なIDでした。過去の9B使用記録は履歴として残っています。短い生成の検証結果は [JSON記録](thermal-27b-routing-2026-10-10.json) に保存しています。

長時間の生成速度、最大出力49,152トークン、画像入力そのものの生成は今回の短い確認では検証していません。

## legacy整理と遅延の再確認

同日の追加確認で `lmstudio-native` を削除し、LM Studioの接続を `lmstudio` に統一しました。モデル一覧の27Bと9B、コンテキストと出力上限を保持し、旧providerの要約先・モデルポリシーも統合しました。LM Studioの要約先は27Bへ統一し、Unslothのモデル別要約設定を保持します。インストール済みDSHの圧縮処理は新しい選択より前のrequest/headerを参照するため、9B用の個別要約先を設定すると27B選択後にも9Bを呼ぶ症状が再発します。現在選ばれているUnslothの既定モデルは変更していません。

`configure_harness_gateway.cjs --routes-only` はDSH設定だけを変更します。GPU設定、共通サービスの設定、認証情報には触れません。変更前に実行中のDSHターンがないことと、ファイルが確認時から変わっていないことを確認します。バックアップはユーザーディレクトリの `route-backups` に保存します。過去のlegacy利用履歴は書き換えません。次回選択が旧providerのままの場合は、DSHのモデル選択から `lmstudio` の同じモデルへ変更してから実行してください。

遅いと報告された新しい会話では、27Bのrequest/headerから最初のストリームまで107,316 msでした。その後も推論・ツール引数のストリームが届いており、モデルが完全に停止していた記録ではありません。ユーザーが356,736 ms後に中断しています。会話本文や推論内容はこの記録へ含めていません。

整理後に `validate_harness_gateway.cjs --provider lmstudio --model qwen/qwen3.8-27b` を再実行し、実DSH LlmRuntimeから共通8877経由で27BのQ4_K_Mファイル、回答42、正常stopを確認しました。初回ストリーム32,550 ms、完了49,587 msでした。設定上の最大出力49,152に対し、この検証の出力上限は128です。

この測定はP100高温時のRTX 4070 + CPU構成です。P100には別のPythonプロセスが約16GBを確保しており、通常構成の性能測定とは扱いません。27BのGPU配置は20層で、CPU側の生成と長い入力の処理が遅延に関係する可能性があります。モデル量子化、KV型、コンテキスト、思考設定を下げる変更は行っていません。

今回の再確認でPythonテスト150件中148件成功・2件skip、legacy整理のNode回帰テスト3件成功、関連コードとテストのty・Ruff・Node構文・PowerShell構文検査を行っています。導入済み共通runtimeの各ファイルとSHA-256管理記録はソースと一致しました。Unsloth 27Bの実温度による退避と復帰は上記の同日検証を参照し、この追加確認では冷却復帰を再測定していません。

このコミットには、温度切替が使う共有サービス・会話圧縮・履歴保存の未追跡だった基盤と、その導入ツール・回帰テストを含めます。Memorixワーカーの導入変更、過去の速度調整、生成された配布物、個人設定・認証情報・会話ログは今回のコミット範囲から除外します。

インストール済み `BasicCompactionEngine.summarize` に前回9Bのrequest/headerと次回27Bの設定を与える回帰確認も行いました。要約の生成自体はモックですが、モデル選択は実DSHコードを実行し、9Bへ誤って戻る失敗を再現した後、27Bへ固定されたことを確認しました。コミット予定のファイルだけを一時ディレクトリへ取り出して実行したPythonテスト115件も成功しました。

検証途中で短い要求が1回、回答が空の状態で失敗しました。この要求のHTTP状態とfinishを保存していなかったため、原因の断定はできません。背景要約をキャンセルした直後にnativeスロットがまだ使用中の場合は、スロット解放を確認する待機を追加しました。待機終了後も使用中なら生成を止めずにエラーを返します。競合を想定した2件の回帰テストと、修正を配布した後の実DSHリクエストが成功しました。最終確認はHTTP 200、27B実ファイル、回答42、正常stop、初回4,705 ms・完了19,916 msでした。

更新中、P100のVRAMが他のPythonプロセスで満杯の状態では、更新処理の `llama-server --list-devices` が終了コード3221226505で失敗しました。初回導入時だけCUDAのデバイスを調べ、既存ヘルパーの更新には保存された構成とハッシュを使うように変更しました。GPU占有を模擬した失敗をテストで再現した後、実機の更新も成功しています。他のPythonプロセスは停止していません。
