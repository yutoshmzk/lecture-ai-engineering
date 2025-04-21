import os
import torch
from transformers import pipeline
import time
import traceback
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import nest_asyncio
from pyngrok import ngrok
import logging

# --- 設定 ---

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 設定 ---
# 環境変数からモデル名を取得、なければデフォルト値を使用
DEFAULT_MODEL_NAME = "google/gemma-2-2b-jpn-it"
MODEL_NAME = os.environ.get("LLM_MODEL_NAME", DEFAULT_MODEL_NAME)
logger.info(f"使用するモデル名: {MODEL_NAME}")

# # モデル名を設定
# MODEL_NAME = "google/gemma-2-2b-jpn-it"  # お好みのモデルに変更可能です
# print(f"モデル名を設定: {MODEL_NAME}")

# # --- モデル設定クラス ---
# class Config:
#     def __init__(self, model_name=MODEL_NAME):
#         self.MODEL_NAME = model_name

# config = Config(MODEL_NAME)

# --- FastAPIアプリケーション定義 ---
app = FastAPI(
    title="ローカルLLM APIサービス",
    description="transformersモデルを使用したテキスト生成のためのAPI",
    version="1.0.0"
)
# app.state にモデルパイプラインを格納するための属性を初期化
app.state.model_pipeline = None

# CORSミドルウェアを追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- データモデル定義 ---
class Message(BaseModel):
    role: str
    content: str

# 直接プロンプトを使用した簡略化されたリクエスト
class SimpleGenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 512
    do_sample: Optional[bool] = True
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class GenerationResponse(BaseModel):
    generated_text: str
    response_time: float

# --- モデル関連の関数 ---
def load_model(model_name: str) -> Optional[Pipeline]:
    """推論用のLLMモデルを読み込む"""
    try:
        # デバイス選択の改善 (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available(): # MPS サポートを追加
            device = "mps"
            # MPS 使用時の注意: bfloat16 がサポートされていない場合がある
            # 必要に応じて torch_dtype を torch.float16 や None に変更
            logger.info("MPSデバイスが利用可能です。torch_dtype=torch.bfloat16 を使用します。")
        else:
            device = "cpu"
        logger.info(f"使用デバイス: {device}")

        # モデルロード中のメッセージ
        logger.info(f"モデル '{model_name}' の読み込みを開始します...")
        start_load_time = time.time()

        pipe = pipeline(
            "text-generation",
            model=model_name,
            # MPSの場合、bfloat16がエラーになる場合は float16 を試す
            model_kwargs={"torch_dtype": torch.bfloat16 if device != "mps" else torch.float16},
            device=device
        )
        end_load_time = time.time()
        logger.info(f"モデル '{model_name}' の読み込みに成功しました (所要時間: {end_load_time - start_load_time:.2f}秒)")
        return pipe
    except ImportError as e:
        logger.error(f"必要なライブラリが見つかりません: {e}. transformers, torch, accelerate などがインストールされているか確認してください。", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"モデル '{model_name}' の読み込み中に予期せぬエラーが発生しました: {e}", exc_info=True) # トレースバックもログに出力
        return None

# --- 依存性注入 ---
# モデルパイプラインを取得するための依存性注入関数
async def get_model_pipeline(request: Request) -> Pipeline:
    if request.app.state.model_pipeline is None:
        logger.error("モデルパイプラインがロードされていません。リクエストを処理できません。")
        raise HTTPException(status_code=503, detail="モデルが利用できません。サーバーが初期化中か、ロードに失敗した可能性があります。")
    return request.app.state.model_pipeline

# 型ヒント付きの依存性定義
ModelPipelineDep = Annotated[Pipeline, Depends(get_model_pipeline)]

# --- FastAPIエンドポイント定義 ---
@app.on_event("startup")
async def startup_event():
    """起動時にモデルを初期化し、app.state に格納"""
    logger.info("アプリケーション起動イベント: モデルの読み込みを開始...")
    pipeline_instance = load_model(MODEL_NAME)
    if pipeline_instance:
        app.state.model_pipeline = pipeline_instance
        logger.info("モデルの初期化が完了し、APIリクエストの準備ができました。")
    else:
        logger.error("起動時のモデル初期化に失敗しました。APIは起動しますが、/generate エンドポイントは機能しません。")

@app.get("/")
async def root():
    """基本的なAPIチェック用のルートエンドポイント"""
    return {"status": "ok", "message": "Local LLM API is running"}

@app.get("/health")
async def health_check(request: Request):
    """ヘルスチェックエンドポイント"""
    model_loaded = request.app.state.model_pipeline is not None
    status = "ok" if model_loaded else "error"
    message = "Model loaded successfully" if model_loaded else "Model not loaded or failed to load"
    return {"status": status, "message": message, "model_name": MODEL_NAME if model_loaded else None}

# 簡略化されたエンドポイント (依存性注入を使用)
@app.post("/generate", response_model=GenerationResponse)
async def generate_simple(request: SimpleGenerationRequest, model_pipeline: ModelPipelineDep):
    """単純なプロンプト入力に基づいてテキストを生成 (依存性注入を使用)"""
    try:
        start_time = time.time()
        # プロンプトが長い場合、ログには一部のみ表示
        log_prompt = request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
        logger.info(f"リクエスト受信: prompt='{log_prompt}', max_new_tokens={request.max_new_tokens}")

        # 依存性注入で取得したモデルパイプラインを使用
        logger.info("モデル推論を開始...")
        outputs = model_pipeline(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            # Gemma-it 用の停止シーケンスを追加 (任意)
            # eos_token_id=model_pipeline.tokenizer.eos_token_id,
            # pad_token_id=model_pipeline.tokenizer.pad_token_id # pad_token_id が設定されている場合
        )
        logger.info("モデル推論が完了しました。")

        # アシスタント応答を抽出
        assistant_response = extract_assistant_response(outputs, request.prompt)
        log_response = assistant_response[:100] + "..." if len(assistant_response) > 100 else assistant_response
        logger.info(f"抽出された応答: '{log_response}'")

        end_time = time.time()
        response_time = end_time - start_time
        logger.info(f"応答生成時間: {response_time:.4f}秒")

        return GenerationResponse(
            generated_text=assistant_response,
            response_time=response_time
        )

    except Exception as e:
        logger.error(f"応答生成中に予期せぬエラーが発生しました: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"応答の生成中に内部サーバーエラーが発生しました。")

# --- モデル関連の関数 ---
# モデルのグローバル変数
# model = None

# def load_model():
#     """推論用のLLMモデルを読み込む"""
#     global model  # グローバル変数を更新するために必要
#     try:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"使用デバイス: {device}")
#         pipe = pipeline(
#             "text-generation",
#             model=config.MODEL_NAME,
#             model_kwargs={"torch_dtype": torch.bfloat16},
#             device=device
#         )
#         print(f"モデル '{config.MODEL_NAME}' の読み込みに成功しました")
#         model = pipe  # グローバル変数を更新
#         return pipe
#     except Exception as e:
#         error_msg = f"モデル '{config.MODEL_NAME}' の読み込みに失敗: {e}"
#         print(error_msg)
#         traceback.print_exc()  # 詳細なエラー情報を出力
#         return None

def extract_assistant_response(outputs: List[Dict[str, Any]], user_prompt: str) -> str:
    """
    モデルの出力からアシスタントの応答を抽出する。
    Gemma-it の一般的な出力形式を想定。
    """
    assistant_response = ""
    try:
        if not outputs or not isinstance(outputs, list) or len(outputs) == 0:
            logger.warning("モデルからの出力が空または予期しない形式です。")
            return "応答を生成できませんでした。"

        # 通常、transformers pipeline はリストの最初の要素に結果を格納する
        generated_output = outputs[0].get("generated_text")

        if not generated_output or not isinstance(generated_output, str):
            logger.warning(f"出力から 'generated_text' を取得できないか、文字列ではありません: {outputs[0]}")
            return "応答の形式が正しくありません。"

        full_text = generated_output.strip()

        # Gemma Instructモデルは通常、入力プロンプトを含まずに応答のみを返す
        # もしプロンプトが含まれる場合は、削除するロジックが必要になる場合がある
        # 例: if full_text.startswith(user_prompt):
        #         assistant_response = full_text[len(user_prompt):].strip()
        #     else:
        #         assistant_response = full_text
        # ここではプロンプトが含まれないと仮定
        assistant_response = full_text

        # 特定のモデル（例：チャット形式にファインチューニングされたモデル）が出力に
        # <start_of_turn>model\n のようなマーカーを含む場合、それを取り除く処理を追加可能
        # model_turn_start = "<start_of_turn>model\n"
        # start_index = assistant_response.find(model_turn_start)
        # if start_index != -1:
        #     assistant_response = assistant_response[start_index + len(model_turn_start):].strip()

        # <end_of_turn> などの終了マーカーがあれば削除
        # end_turn = "<end_of_turn>"
        # if assistant_response.endswith(end_turn):
        #     assistant_response = assistant_response[:-len(end_turn)].strip()

    except Exception as e:
        logger.error(f"応答の抽出中にエラーが発生しました: {e}", exc_info=True)
        assistant_response = "応答の抽出処理中にエラーが発生しました。"

    if not assistant_response:
        logger.warning(f"アシスタントの応答を抽出できませんでした。モデル出力: {outputs}")
        assistant_response = "応答を抽出できませんでした。"

    return assistant_response

# def extract_assistant_response(outputs, user_prompt):
#     """モデルの出力からアシスタントの応答を抽出する"""
#     assistant_response = ""
#     try:
#         if outputs and isinstance(outputs, list) and len(outputs) > 0 and outputs[0].get("generated_text"):
#             generated_output = outputs[0]["generated_text"]
            
#             if isinstance(generated_output, list):
#                 # メッセージフォーマットの場合
#                 if len(generated_output) > 0:
#                     last_message = generated_output[-1]
#                     if isinstance(last_message, dict) and last_message.get("role") == "assistant":
#                         assistant_response = last_message.get("content", "").strip()
#                     else:
#                         # 予期しないリスト形式の場合は最後の要素を文字列として試行
#                         print(f"警告: 最後のメッセージの形式が予期しないリスト形式です: {last_message}")
#                         assistant_response = str(last_message).strip()

#             elif isinstance(generated_output, str):
#                 # 文字列形式の場合
#                 full_text = generated_output
                
#                 # 単純なプロンプト入力の場合、プロンプト後の全てを抽出
#                 if user_prompt:
#                     prompt_end_index = full_text.find(user_prompt)
#                     if prompt_end_index != -1:
#                         prompt_end_pos = prompt_end_index + len(user_prompt)
#                         assistant_response = full_text[prompt_end_pos:].strip()
#                     else:
#                         # 元のプロンプトが見つからない場合は、生成されたテキストをそのまま返す
#                         assistant_response = full_text
#                 else:
#                     assistant_response = full_text
#             else:
#                 print(f"警告: 予期しない出力タイプ: {type(generated_output)}")
#                 assistant_response = str(generated_output).strip()  # 文字列に変換

#     except Exception as e:
#         print(f"応答の抽出中にエラーが発生しました: {e}")
#         traceback.print_exc()
#         assistant_response = "応答の抽出に失敗しました。"  # エラーメッセージを設定

#     if not assistant_response:
#         print("警告: アシスタントの応答を抽出できませんでした。完全な出力:", outputs)
#         # デフォルトまたはエラー応答を返す
#         assistant_response = "応答を生成できませんでした。"

#     return assistant_response

# --- FastAPIエンドポイント定義 ---
# @app.on_event("startup")
# async def startup_event():
#     """起動時にモデルを初期化"""
#     load_model_task()  # バックグラウンドではなく同期的に読み込む
#     if model is None:
#         print("警告: 起動時にモデルの初期化に失敗しました")
#     else:
#         print("起動時にモデルの初期化が完了しました。")

# @app.get("/")
# async def root():
#     """基本的なAPIチェック用のルートエンドポイント"""
#     return {"status": "ok", "message": "Local LLM API is runnning"}

# @app.get("/health")
# async def health_check():
#     """ヘルスチェックエンドポイント"""
#     global model
#     if model is None:
#         return {"status": "error", "message": "No model loaded"}

#     return {"status": "ok", "model": config.MODEL_NAME}

# 簡略化されたエンドポイント
# @app.post("/generate", response_model=GenerationResponse)
# async def generate_simple(request: SimpleGenerationRequest):
#     """単純なプロンプト入力に基づいてテキストを生成"""
#     global model

#     if model is None:
#         print("generateエンドポイント: モデルが読み込まれていません。読み込みを試みます...")
#         load_model_task()  # 再度読み込みを試みる
#         if model is None:
#             print("generateエンドポイント: モデルの読み込みに失敗しました。")
#             raise HTTPException(status_code=503, detail="モデルが利用できません。後でもう一度お試しください。")

#     try:
#         start_time = time.time()
#         print(f"シンプルなリクエストを受信: prompt={request.prompt[:100]}..., max_new_tokens={request.max_new_tokens}")  # 長いプロンプトは切り捨て

#         # プロンプトテキストで直接応答を生成
#         print("モデル推論を開始...")
#         outputs = model(
#             request.prompt,
#             max_new_tokens=request.max_new_tokens,
#             do_sample=request.do_sample,
#             temperature=request.temperature,
#             top_p=request.top_p,
#         )
#         print("モデル推論が完了しました。")

#         # アシスタント応答を抽出
#         assistant_response = extract_assistant_response(outputs, request.prompt)
#         print(f"抽出されたアシスタント応答: {assistant_response[:100]}...")  # 長い場合は切り捨て

#         end_time = time.time()
#         response_time = end_time - start_time
#         print(f"応答生成時間: {response_time:.2f}秒")

#         return GenerationResponse(
#             generated_text=assistant_response,
#             response_time=response_time
#         )

#     except Exception as e:
#         print(f"シンプル応答生成中にエラーが発生しました: {e}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"応答の生成中にエラーが発生しました: {str(e)}")

# def load_model_task():
#     """モデルを読み込むバックグラウンドタスク"""
#     global model
#     print("load_model_task: モデルの読み込みを開始...")
#     # load_model関数を呼び出し、結果をグローバル変数に設定
#     loaded_pipe = load_model()
#     if loaded_pipe:
#         model = loaded_pipe  # グローバル変数を更新
#         print("load_model_task: モデルの読み込みが完了しました。")
#     else:
#         print("load_model_task: モデルの読み込みに失敗しました。")

# print("FastAPIエンドポイントを定義しました。")

# --- ngrokでAPIサーバーを実行する関数 ---
def run_with_ngrok(port=8501):
    """ngrokでFastAPIアプリを実行"""
    nest_asyncio.apply() # Colab/Jupyter環境でasyncioイベントループをネスト可能にする

    ngrok_token = os.environ.get("NGROK_TOKEN")
    if not ngrok_token:
        logger.error("Ngrok認証トークンが環境変数 'NGROK_TOKEN' に設定されていません。")
        logger.info("Ngrokを使用するには、https://dashboard.ngrok.com/get-started/your-authtoken でトークンを取得し、")
        logger.info("環境変数 'NGROK_TOKEN' に設定してください（例: Colab Secrets）。")
        # 対話的な入力はサーバー環境で問題を起こすため削除
        # try:
        #     ngrok_token = input("Ngrok認証トークンを入力してください: ")
        # except EOFError:
        #     logger.error("対話型入力が利用できません。環境変数を設定してください。")
        #     return
        return # トークンがない場合は終了

    try:
        ngrok.set_auth_token(ngrok_token)
        logger.info("Ngrok認証トークンを設定しました。")

        # 既存のngrokトンネルを閉じる試み
        try:
            tunnels = ngrok.get_tunnels()
            if tunnels:
                logger.info(f"{len(tunnels)}個の既存ngrokトンネルが見つかりました。閉じています...")
                for tunnel in tunnels:
                    try:
                        ngrok.disconnect(tunnel.public_url)
                        logger.info(f"  - トンネル切断: {tunnel.public_url}")
                    except Exception as disconnect_err:
                        logger.warning(f"  - トンネル切断中にエラー: {tunnel.public_url} - {disconnect_err}")
                logger.info("既存ngrokトンネルの切断処理を完了しました。")
            else:
                logger.info("アクティブなngrokトンネルはありません。")
        except Exception as e:
            logger.warning(f"既存トンネルの取得・切断中にエラーが発生しました: {e}", exc_info=True)

        # 新しいngrokトンネルを開く
        logger.info(f"ポート {port} で新しいngrokトンネルを開いています...")
        # プロトコルを明示的に指定 (通常は 'http')
        public_url = ngrok.connect(port, "http").public_url
        logger.info("---------------------------------------------------------------------")
        logger.info(f"✅ FastAPI アプリケーションが公開されました！")
        logger.info(f"✅ 公開URL (Public URL): {public_url}")
        logger.info(f"📖 APIドキュメント (Swagger UI): {public_url}/docs")
        logger.info(f"🩺 ヘルスチェック: {public_url}/health")
        logger.info("---------------------------------------------------------------------")
        logger.info("(Ctrl+C または セルを停止 するとサーバーとトンネルが終了します)")

        # Uvicornサーバーの起動設定
        # log_config=None で uvicorn のデフォルトロガー設定を無効化し、
        # Python標準のlogging設定に統一する（任意）
        uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info", log_config=None)
        server = uvicorn.Server(uvicorn_config)

        # Uvicornサーバーを実行 (非同期的に実行される)
        server.run()
        # server.run() が終了したら (通常は Ctrl+C などで停止された場合)
        logger.info("Uvicornサーバーが停止しました。")

    except Exception as e:
        logger.error(f"ngrokまたはUvicornの起動/実行中にエラーが発生しました: {e}", exc_info=True)
    finally:
        # アプリケーション終了時にngrokトンネルを閉じる
        try:
            logger.info("ngrokトンネルを閉じています...")
            ngrok.kill() # すべてのngrokプロセスを停止
            logger.info("ngrokトンネルを閉じました。")
        except Exception as e:
            logger.warning(f"ngrokトンネルのクリーンアップ中にエラーが発生しました: {e}", exc_info=True)


# --- ngrokでAPIサーバーを実行する関数 ---
# def run_with_ngrok(port=8501):
#     """ngrokでFastAPIアプリを実行"""
#     nest_asyncio.apply()

#     ngrok_token = os.environ.get("NGROK_TOKEN")
#     if not ngrok_token:
#         print("Ngrok認証トークンが'NGROK_TOKEN'環境変数に設定されていません。")
#         try:
#             print("Colab Secrets(左側の鍵アイコン)で'NGROK_TOKEN'を設定することをお勧めします。")
#             ngrok_token = input("Ngrok認証トークンを入力してください (https://dashboard.ngrok.com/get-started/your-authtoken): ")
#         except EOFError:
#             print("\nエラー: 対話型入力が利用できません。")
#             print("Colab Secretsを使用するか、ノートブックセルで`os.environ['NGROK_TOKEN'] = 'あなたのトークン'`でトークンを設定してください")
#             return

#     if not ngrok_token:
#         print("エラー: Ngrok認証トークンを取得できませんでした。中止します。")
#         return

#     try:
#         ngrok.set_auth_token(ngrok_token)

#         # 既存のngrokトンネルを閉じる
#         try:
#             tunnels = ngrok.get_tunnels()
#             if tunnels:
#                 print(f"{len(tunnels)}個の既存トンネルが見つかりました。閉じています...")
#                 for tunnel in tunnels:
#                     print(f"  - 切断中: {tunnel.public_url}")
#                     ngrok.disconnect(tunnel.public_url)
#                 print("すべての既存ngrokトンネルを切断しました。")
#             else:
#                 print("アクティブなngrokトンネルはありません。")
#         except Exception as e:
#             print(f"トンネル切断中にエラーが発生しました: {e}")
#             # エラーにもかかわらず続行を試みる

#         # 新しいngrokトンネルを開く
#         print(f"ポート{port}に新しいngrokトンネルを開いています...")
#         ngrok_tunnel = ngrok.connect(port)
#         public_url = ngrok_tunnel.public_url
#         print("---------------------------------------------------------------------")
#         print(f"✅ 公開URL:   {public_url}")
#         print(f"📖 APIドキュメント (Swagger UI): {public_url}/docs")
#         print("---------------------------------------------------------------------")
#         print("(APIクライアントやブラウザからアクセスするためにこのURLをコピーしてください)")
#         uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")  # ログレベルをinfoに設定

#     except Exception as e:
#         print(f"\n ngrokまたはUvicornの起動中にエラーが発生しました: {e}")
#         traceback.print_exc()
#         # エラー後に残る可能性のあるngrokトンネルを閉じようとする
#         try:
#             print("エラーにより残っている可能性のあるngrokトンネルを閉じています...")
#             tunnels = ngrok.get_tunnels()
#             for tunnel in tunnels:
#                 ngrok.disconnect(tunnel.public_url)
#             print("ngrokトンネルを閉じました。")
#         except Exception as ne:
#             print(f"ngrokトンネルのクリーンアップ中に別のエラーが発生しました: {ne}")

# --- メイン実行ブロック ---
if __name__ == "__main__":
    logger.info("メイン実行ブロック: FastAPIアプリケーションサーバーを開始します...")
    # 指定されたポートでサーバーを起動
    run_with_ngrok(port=8501) # ポート番号は必要に応じて変更
    logger.info("サーバープロセスが正常に終了しました。")

# # --- メイン実行ブロック ---
# if __name__ == "__main__":
#     # 指定されたポートでサーバーを起動
#     run_with_ngrok(port=8501)  # このポート番号を確認
#     # run_with_ngrokが終了したときにメッセージを表示
#     print("\nサーバープロセスが終了しました。")