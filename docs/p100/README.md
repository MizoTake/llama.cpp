# P100優先ビルドの運用

このforkでは、Windows x64 / Ryzen 5 7600 / Tesla P100 16GB / RTX 4070 12GB向けのビルドと起動設定を管理します。C++ソースやCUDAカーネルの変更はありません。

今回の共通サービス・Unsloth 27B温度切替・DSHのlegacy整理と遅延確認は [運用記録](thermal-27b-routing-2026-10-10.md) を参照してください。P100優先の両GPU構成、高温時のRTX + CPU退避、区切りでの復帰、会話圧縮・共有メモリを管理します。DSHは `lmstudio` / `unslothdesktop` に統一し、各モデルの要約先を明示します。

2026-10-11に、両アプリの9B/27Bへ上限8 GiBの [RAMプロンプトキャッシュ](ram-cache-2026-10-11.md) を明示しました。RTX退避構成で別の会話を挟んで4,846トークンの入力へ戻る測定では4,842トークンを復元し、最初の出力までの時間が無効時より約89-94%短くなりました。LM Studioは元から初期値で有効で、UnslothはWindowsの全層GPU配置時に自動無効化される場合があります。新規入力の高速化は確認していません。測定条件、チェックポイントの負担、再適用方法はリンク先を参照してください。

P100優先の通常構成でも、両アプリの9B/27BでRAM無効・有効を [追加比較](ram-cache-primary-2026-10-11.md) しています。通常のデバイス順、3:2配分、F16 KV、各モデルのコンテキストを維持し、温度境界に達した測定は採用しません。短い入力・各1回の観測として、RTXの長文測定とは分けて記録します。

## 最初のビルドで行ったこと

基準のソースは `11fe02151f79c41d0d4af7da708755d73b9c0da6` です。既存のMSVC 19.44、CMake 4.1.2、Ninja、CUDA Toolkit 12.9.86、OpenSSL 3.6.0を使用しました。Python依存は追加せず、既存のUnsloth環境を使いました。

| 設定 | 目的 |
| --- | --- |
| `CMAKE_BUILD_TYPE=Release` | リリース向けのコンパイラー最適化 |
| `GGML_CUDA=ON` | CUDAによるGPU推論 |
| `CMAKE_CUDA_ARCHITECTURES=60-real;89-real` | P100とRTX 4070用のGPU機械語を両方収録 |
| `GGML_NATIVE=ON` | ビルドしたPCのCPU機能を検出。最初のビルドでは `/arch:AVX512` を確認 |
| `GGML_LTO=ON` | リンク時最適化。MSVCの `/GL` を確認 |
| `GGML_CUDA_FA_QUANTS=all` | Flash AttentionのK/V型の組み合わせを収録。コンパイル時間は増える |
| `BUILD_SHARED_LIBS=OFF` | llama/ggmlを実行ファイルへ静的リンク。CUDA/OpenSSLのDLLは別途同梱 |

P100はPascal世代です。CUDA 13ではPascal向けのオフラインコンパイルとライブラリー対応が削除されているため、このプリセットはCUDA 12.9を使用します。[NVIDIAのリリースノート](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)

GPU配分は既存のlayer splitを使います。P100:RTX 4070を2:1と3:2で測定し、今回の9Bモデルでは3:2の方が生成速度が高かったため、3:2を既定値にしました。配分はレイヤー単位なので、使用メモリーが厳密に60%/40%になるわけではありません。

これは対象PC用のビルド設定と実測による実行設定です。未最適化ビルドとの同条件比較はしていないため、コンパイル設定による速度向上率は未測定です。変換と量子化は主にCPUを使用します。

## Gitで管理するもの

- `CMakePresets.json`: `p100-rtx4070-release` プリセット。
- `scripts/p100/`: ビルド、配布、起動、Unsloth設定と復元のスクリプト。
- `docs/p100/`: この手順と最初の検証記録。

`build-p100-rtx4070/`、`release/`、モデル、UnslothのDBや個人設定バックアップはGit管理外です。Gitで管理したソースと手順から配布物を作り、ZIPが必要な場合は別の配布ストレージやforkのGitHub Releasesへ保管します。

## 再ビルドと配布

以下はリポジトリのルートで実行します。既存のWindows C++用Visual Studio、CMake、CUDA 12.9、OpenSSLが必要です。スクリプトは依存をインストールしません。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\p100\build.ps1 -Tests
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\p100\package.ps1 -Zip
```

`build.ps1` はプリセットを使って7ツールをビルドし、`-Tests` を指定すると選択したテスト用ターゲットをビルドしてCTestを実行します。GPU演算テストの実行、変換、Unslothモデル読み込みはこのコマンドには含みません。CUDAの配置を変えた場合は `-CudaRoot`、並列数は `-Parallel` で指定できます。

このPCで管理用スクリプトを実行した際は35/35件が通過しました。最初の34件に、既存Python環境のJinja2を使う `test-jinja-py` を含めた結果です。`-Tests` ではこのPython環境も必要です。

`package.ps1` はEXE、CUDA/OpenSSL DLL、HF/LoRA変換ツール、起動・設定スクリプト、ライセンスを新しい `release/p100-first-日時/` へ収録します。既存の配布フォルダーは上書きしません。OpenSSLの配置を変えた場合は `-OpenSslRoot` で指定します。

配布物の `build-manifest.json` はソースコミット、未コミット変更の有無、ビルド設定、レシピのSHA-256を記録します。各ファイルのSHA-256は `SHA256SUMS.txt`、ZIPのSHA-256は `.zip.sha256` に出力します。未コミット変更がある配布物はコミット番号だけでは再現できないため、継続運用では変更をGitに保存してからビルドします。

## UnslothDesktopで使う

`RUNTIME_DIRECTORY` は `package.ps1` が表示した配布フォルダーです。最初の配布物はリポジトリ内の `release/p100-first` にあります。以下のスクリプトをリポジトリ側から実行する場合、省略時はこの最初の配布物を使います。新しい配布物へ切り替える際は必ず場所を指定します。

```powershell
& "$env:USERPROFILE\.unsloth\studio\unsloth_studio\Scripts\python.exe" -X utf8 .\scripts\p100\configure_unsloth.py --runtime "RUNTIME_DIRECTORY" --dry-run
& "$env:USERPROFILE\.unsloth\studio\unsloth_studio\Scripts\python.exe" -X utf8 .\scripts\p100\configure_unsloth.py --runtime "RUNTIME_DIRECTORY"
```

`--dry-run` は変更予定を表示し、設定を保存しません。保存時はカスタムllama.cppパスとモデル設定を `~/.unsloth/studio/backups/llama-p100/` にバックアップしてから、インストール済みUnslothの設定APIを使って変更します。Unsloth更新で内部APIが変わった場合はスクリプトも確認してください。

保存済みGGUF設定と、LM Studio/Hugging Faceキャッシュ内のトークナイザーを持つ言語GGUFを対象にします。Hugging FaceモデルはリポジトリIDと量子化形式の組み合わせでも保存し、Desktopのモデル選択から参照できるようにします。既存の追加引数のうち、このプリセットが管理するsplit/Flash Attention/KV型/batch以外は保持します。SM 6.0/8.9のGPUを検出し、P100の番号を固定せずP100を先頭にして重み3、他GPUへ重み2を割り当てます。

このPCの通常の物理番号はRTX 4070が0、P100が1なので、保存される設定はGPU `[1,0]` と `--split-mode layer --tensor-split 3,2 --flash-attn on` です。KVはF16、Tensor Parallelismは無効、GPUメモリーはManual、GPU layersは1024を保存します。1024は全レイヤーをGPUへ載せるための上限値です。VRAMに収まらないモデルやコンテキスト長は手動で調整します。

batch/ubatchとコンテキスト長は保存値を保持します。未設定ならbatch/ubatchは512、コンテキスト長は8,192、同時応答数は1です。`--model "MODEL_ID"` で対象を選択でき、複数指定も可能です。`--batch-size`、`--ubatch-size`、`--context-length` は指定した項目だけを更新し、範囲外の値や見つからない対象ID、ubatchがbatchより大きい設定は保存前に拒否します。

UnslothDesktopを終了して再起動してください。変換ツールの場所も環境変数で渡して起動する場合は、終了後に次を実行します。モデル側の保存設定は起動時の環境変数より優先されます。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\p100\start-unsloth-p100.ps1 -RuntimeDir "RUNTIME_DIRECTORY"
```

復元する場合はアプリを終了し、設定時に表示されたバックアップを指定します。復元はこの2設定全体を戻すため、その後に追加したモデル設定も変更前へ戻ります。

```powershell
& "$env:USERPROFILE\.unsloth\studio\unsloth_studio\Scripts\python.exe" -X utf8 .\scripts\p100\restore_unsloth.py "BACKUP_FILE.json"
```

## 単独実行と変換

単独起動スクリプトもP100をCUDA0に並べ、配分を `3,2` にします。Unslothは物理番号 `[1,0]` を使い、同じP100優先の配分を保存します。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\p100\run-llama-p100.ps1 -BinaryDir "RUNTIME_DIRECTORY" -Tool server -m "MODEL.gguf" -c 8192 -ngl 99 -fa auto --host 127.0.0.1 --port 8081
& "$env:USERPROFILE\.unsloth\studio\unsloth_studio\Scripts\python.exe" -X utf8 "RUNTIME_DIRECTORY\convert_hf_to_gguf.py" "HF_MODEL_DIRECTORY" --outfile "model-f16.gguf" --outtype f16
& "RUNTIME_DIRECTORY\llama-quantize.exe" "model-f16.gguf" "model-Q4_K_M.gguf" Q4_K_M 6
```

`-Tool` はserver、cli、bench、quantize、imatrix、gguf-split、perplexityを選べます。`-P100Only` はGPU単独で切り分ける場合に使います。

## 最初の検証と更新時の確認

[検証記録](validation-2026-10-04.json) は基準コミットでの結果です。後のコミットや別のPCの検証結果には流用しません。

- 選択した既存CTest: 34/34 PASS。
- F16/Q4_K行列演算とCPU結果の比較: 各GPU 26/26 PASS。
- F16 Flash AttentionとCPU結果の比較: 各GPU 4/4 PASS。
- 小型HF GPT2からF16 GGUFへの変換とQ4_K_M量子化: PASS。
- インストール済みUnslothバックエンドによる両GPUモデル読み込みとチャット応答: PASS。
- UnslothDesktopのGUI操作と大規模HFモデルの全量変換: 未検証。

全ターゲットのビルドは既存の `tests/test-batch-alloc.cpp:785` のMSVC C2131で停止しました。上記34件にこのテストは含まれません。WindowsではNCCLが見つからない警告も出ています。標準のCUDA layer splitによる両GPU利用を検証しました。

upstream更新時は通常のGit mergeで取り込み、この手順で再ビルドします。CTestに加え、`llama-server --list-devices`、GPU演算テスト、使うモデルのbench、HF変換・量子化、Unslothでのモデル読み込みを確認します。新しいアプリやCUDAへの切り替えだけでP100対応が維持されるとは判断しません。

ベンチマークはQ4_K_M、F16 KV、プロンプト128、生成32、3回、6スレッド、batch 512、ubatch 128で測定しました。9BはP100:RTX 4070 = 3:2 / Flash Attention onで生成44.47 tokens/s、27Bは同じ配分 / autoで14.98 tokens/sです。長いコンテキストや別モデルの最適比率は未検証です。

[追加測定の記録](tuning-2026-10-04.md)ではスレッド数を比較しました。持続負荷でP100の温度制限が出たため、GPU配分・バッチの最適化は冷却条件を確認してから再測定します。
