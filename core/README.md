# GraphRAG AWS Bedrock Core Implementation

修正版AWS Bedrock統合実装（Novaモデルを除く）

## 🔧 修正された問題

### 1. **API仕様の修正**
- ✅ Anthropic Claude の正しい `anthropic_version` パラメータ追加
- ✅ `messages` 配列の正しい構造実装
- ✅ `content` フィールドの配列形式対応

### 2. **プロトコル完全実装**
- ✅ `ChatModel` プロトコルに準拠した `ModelResponse` 返却
- ✅ `EmbeddingModel` プロトコルの全メソッド実装
- ✅ 同期・非同期両対応

### 3. **エラーハンドリング強化**
- ✅ AWS ClientError の適切な処理
- ✅ HTTP ステータスコードチェック
- ✅ JSON パースエラー処理
- ✅ カスタム `BedrockAPIError` 例外

### 4. **レスポンス処理修正**
- ✅ Anthropic レスポンス形式の正しいパース
- ✅ Titan 埋め込みモデルの正しい数値配列返却
- ✅ V2モデル対応（dimensions, normalize オプション）

## 📁 ファイル構成

```
core/
├── graphrag_aws/
│   ├── __init__.py              # パッケージエクスポート
│   ├── bedrock_models.py        # 修正済みモデル実装
│   └── factory.py               # ファクトリー（Nova除外）
├── config/
│   └── settings_bedrock_example.yaml  # 設定例
├── tests/
│   └── test_bedrock_models.py   # 単体テスト
└── README.md                    # このファイル
```

## 🚀 使用方法

### 1. AWS認証設定

```bash
# AWS CLI設定
aws configure

# または環境変数
export AWS_REGION=us-east-1
export AWS_PROFILE=your-profile
```

### 2. GraphRAG設定

```yaml
# settings.yaml
llm:
  provider: "bedrock_anthropic_chat"
  model: "anthropic.claude-3-5-sonnet-20241022-v2:0"
  
embeddings:
  provider: "bedrock_text_embedding_v2"
  model: "amazon.titan-embed-text-v2:0"
```

### 3. コードでの使用

```python
from graphrag_aws import BedrockAnthropicChatLLM, BedrockEmbeddingLLM

# チャットモデル
chat_model = BedrockAnthropicChatLLM(
    name="claude",
    config=config
)

response = await chat_model.achat("Hello, how are you?")
print(response.output.content)

# 埋め込みモデル  
embed_model = BedrockEmbeddingLLM(
    name="titan",
    config=config
)

embeddings = await embed_model.aembed_batch([
    "Text to embed 1",
    "Text to embed 2"
])
```

## 🧪 テスト実行

```bash
# 単体テスト実行
cd core
python -m pytest tests/test_bedrock_models.py -v

# カバレッジ確認
python -m pytest tests/test_bedrock_models.py --cov=graphrag_aws --cov-report=html
```

## 📊 サポート対象モデル

### LLMモデル
- ✅ Anthropic Claude 3.5 Sonnet
- ✅ Anthropic Claude 3 Haiku  
- ✅ Anthropic Claude 3 Opus

### 埋め込みモデル
- ✅ Amazon Titan Embed Text v2
- ✅ Amazon Titan Embed Text v1

### 除外されたモデル
- ❌ Amazon Nova（要求により除外）

## 🔧 設定オプション

### Anthropic Claude パラメータ
- `max_tokens`: 最大トークン数（デフォルト: 4096）
- `temperature`: ランダム性（0.0-1.0）
- `top_p`: Top-p サンプリング（0.0-1.0）
- `system`: システムプロンプト（オプション）

### Titan Embedding パラメータ
- `dimensions`: 埋め込み次元数（V2のみ）
- `normalize`: 正規化有無（V2のみ）
- `embeddingTypes`: 埋め込みタイプ（V2のみ）

## 🚨 重要な注意事項

1. **AWS IAMロール権限**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "bedrock:InvokeModel"
         ],
         "Resource": [
           "arn:aws:bedrock:*::foundation-model/anthropic.claude*",
           "arn:aws:bedrock:*::foundation-model/amazon.titan-embed*"
         ]
       }
     ]
   }
   ```

2. **モデルアクセス**
   - AWS Bedrockコンソールでモデルアクセスを有効化
   - 対象リージョンでの利用可能性確認

3. **エラーハンドリング**
   - `BedrockAPIError` をキャッチして適切に処理
   - ネットワークエラーやレート制限に対応

## 🔗 関連リンク

- [AWS Bedrock ドキュメント](https://docs.aws.amazon.com/bedrock/)
- [Anthropic Claude モデル仕様](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html)
- [Titan Embedding モデル仕様](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html)