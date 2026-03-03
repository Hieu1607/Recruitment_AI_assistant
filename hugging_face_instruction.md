PI documentation
hieuailearning/BAAI_bge_m3_api

API Recorder

6 API endpoints


Choose one of the following ways to interact with the API.

1. Install the python client (docs) if you don't already have it installed.

copy
$ pip install gradio_client
2. Find the API endpoint below corresponding to your desired function in the app. Copy the code snippet, replacing the placeholder values with your own input data. If this is a private Space, you may need to pass your Hugging Face token as well (read more). Or use the 
API Recorder

 to automatically generate your API requests.

API name: /get_batch_embeddings Hàm này nhận nhiều chuỗi văn bản (mỗi dòng là một chunk) và trả về vector nhúng (embedding) cho từng chunk.
copy
from gradio_client import Client

client = Client("hieuailearning/BAAI_bge_m3_api")
result = client.predict(
		text_input="Hello!!",
		api_name="/get_batch_embeddings"
)
print(result)
Accepts 1 parameter:
text_input str Required

The input value that is provided in the "Nhiều đoạn văn bản (mỗi dòng là một chunk)" Textbox component.

Returns 1 element
str | float | bool | list | dict

The output value that appears in the "Kết quả Embedding (chi tiết)" Json component.

API name: /lambda
copy
from gradio_client import Client

client = Client("hieuailearning/BAAI_bge_m3_api")
result = client.predict(
		api_name="/lambda"
)
print(result)
Accepts 0 parameters:
Returns 1 element
str

The output value that appears in the "Nhiều đoạn văn bản (mỗi dòng là một chunk)" Textbox component.

API name: /get_embeddings_only Hàm rút gọn chỉ trả về danh sách embeddings.
copy
from gradio_client import Client

client = Client("hieuailearning/BAAI_bge_m3_api")
result = client.predict(
		text_input="Hello!!",
		api_name="/get_embeddings_only"
)
print(result)
Accepts 1 parameter:
text_input str Required

The input value that is provided in the "Văn bản đầu vào (mỗi dòng là một chunk)" Textbox component.

Returns 1 element
str | float | bool | list | dict

The output value that appears in the "Danh sách Embeddings" Json component.

API name: /lambda_1
copy
from gradio_client import Client

client = Client("hieuailearning/BAAI_bge_m3_api")
result = client.predict(
		api_name="/lambda_1"
)
print(result)
Accepts 0 parameters:
Returns 1 element
str

The output value that appears in the "Nhiều đoạn văn bản (mỗi dòng là một chunk)" Textbox component.

API name: /lambda_2
copy
from gradio_client import Client

client = Client("hieuailearning/BAAI_bge_m3_api")
result = client.predict(
		api_name="/lambda_2"
)
print(result)
Accepts 0 parameters:
Returns 1 element
str

The output value that appears in the "Nhiều đoạn văn bản (mỗi dòng là một chunk)" Textbox component.

API name: /lambda_3
copy
from gradio_client import Client

client = Client("hieuailearning/BAAI_bge_m3_api")
result = client.predict(
		api_name="/lambda_3"
)
print(result)
Accepts 0 parameters:
Returns 1 element
str

The output value that appears in the "Nhiều đoạn văn bản (mỗi dòng là một chunk)" Textbox component.