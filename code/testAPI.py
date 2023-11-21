from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os, json, re, io
from os import path
import requests
import mimetypes
import traceback
import chardet
from utilities.helper import LLMHelper
import uuid
import regex as re
from redis.exceptions import ResponseError 
from urllib import parse


import logging
logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)

# using flask_restful 
from flask import Flask, jsonify, request 
from flask_restful import Resource, Api, reqparse 

# creating the flask app 
app = Flask(__name__) 
# creating an API object 
api = Api(app) 

parser = reqparse.RequestParser()


prompt = ""
temperature = 0.7

# making a class for a particular resource 
# the get, post methods correspond to get and post requests 
# they are automatically mapped by flask_restful. 
# other methods include put, delete, etc. 
class Hello(Resource): 

	# corresponds to the GET request. 
	# this function is called whenever there 
	# is a GET request for this resource 
	def get(self): 

		return jsonify({'message': 'hello world'}) 

	# Corresponds to POST request 
	def post(self): 
		
		data = request.get_json()	 # status code 
		return jsonify({'data': data}), 201
 
	
class CheckDeployment(Resource):
	
  def get(self):

    response = {}
		
    try:
        llm_helper = LLMHelper()
        llm_helper.get_completion("Generate a joke!")
        response['llm'] = 'working'
    except Exception as e:
        response['llm'] = 'not working'
    
     #\ 2. Check if the embedding is working
    try:
        llm_helper = LLMHelper()
        llm_helper.embeddings.embed_documents(texts=["This is a test"])
        response['embedding'] = 'working'
    except Exception as e:
        response['embedding'] = 'not working'

    #\ 3. Check if the translation is working
    try:
        llm_helper = LLMHelper()
        llm_helper.translator.translate("This is a test", "it")
        response['translation'] = 'working'
    except Exception as e:
        response['translation'] = 'not working'

        #\ 4. Check if the Redis is working with previous version of data
    try:
        llm_helper = LLMHelper()
        if llm_helper.vector_store_type != "AzureSearch":
            if llm_helper.vector_store.check_existing_index("embeddings-index"):
                response['warning'] = """Seems like you're using a Redis with an old data structure.  
                If you want to use the new data structure, you can start using the app and go to "Add Document" -> "Add documents in Batch" and click on "Convert all files and add embeddings" to reprocess your documents.  
                To remove this working, please delete the index "embeddings-index" from your Redis.  
                If you prefer to use the old data structure, please change your Web App container image to point to the docker image: fruocco/oai-embeddings:2023-03-27_25. 
                """
            else:
                response['redis'] = 'working'
        else:
            try:
                llm_helper.vector_store.index_exists()
                response['Azure Cognitive Search'] = "working"
            except Exception as e:
                response['Azure Cognitive Search'] = 'not working'
    except Exception as e:
        response['redis'] = 'not working'

    return jsonify(response)
  
class GetLanguages(Resource):
    def get(self):
        try:
            llm_helper = LLMHelper()
            return jsonify(llm_helper.translator.get_available_languages())
        except Exception as e:
            return jsonify('no languages found')
        
class Query(Resource):
    def post(self):
        try:
            res = {}
            data = request.get_json()
            if 'prompt' in data:
                llm_helper = LLMHelper(custom_prompt=data["prompt"], temperature=temperature)
            elif "temp" in data:
                print(data["temp"])
                llm_helper = LLMHelper(custom_prompt=prompt, temperature=data["temp"])
            elif "temp" in data and "prompt" in data:
                llm_helper = LLMHelper(custom_prompt = data["prompt"], temperature=data["temp"])
            else:
                llm_helper = LLMHelper(custom_prompt="", temperature=temperature)
            question, response, context, sources = llm_helper.get_semantic_answer_lang_chain(data['question'], [])
            answer, followUps = llm_helper.extract_followupquestions(response)
            res["answer"] = answer
            res["followup_questions"] = followUps
            if sources:
                response, sourceList, matchedSourceList, linkList, fileNames = llm_helper.get_links_filenames(response, sources)
                res["sourceList"] = sourceList
                res["matchedSourceList"] = matchedSourceList
                res["linkList"] = linkList
                res["filenames"] = fileNames
            return jsonify(res)
        except Exception as e:
            return jsonify({'error': f'{e}'})
        
class Chat(Resource):
    def post(self):
        try:
            llm_helper = LLMHelper()
            res = {}
            data = request.get_json()
            chatQuestion = data["chatQuestion"]
            chatHistory = data["chatHistory"]
            chatQuestion, result, context, sources = llm_helper.get_semantic_answer_lang_chain(chatQuestion, chatHistory)
            result, followupQuestions = llm_helper.extract_followupquestions(result)
            chatHistory.append((chatQuestion, result))
            if sources:
                answerWithCitations, sourceList, matchedSourcesList, linkList, filenameList = llm_helper.get_links_filenames(result, sources)
                res["answerWithCitations"] = answerWithCitations
                res["sourceList"] = sourceList
                res["matchedSourceList"] = matchedSourcesList
                res["linkList"] = linkList
                res["filenames"] = filenameList

            res["chatHistory"] = chatHistory
            res["sources"] = sources
            res["context"] = context
            res["result"] = result
            res["followupQuestions"] = followupQuestions
            return jsonify(res)
        except Exception as e:
            return jsonify({'error': f'{e}'})

class AddDocument(Resource):

    def upload_text_and_embeddings():
        file_name = f"{uuid.uuid4()}.txt"
        source_url = llm_helper.blob_client.upload_file(st.session_state['doc_text'], file_name=file_name, content_type='text/plain; charset=utf-8')
        llm_helper.add_embeddings_lc(source_url) 
        st.success("Embeddings added successfully.")

    def remote_convert_files_and_add_embeddings(process_all=False):
        url = os.getenv('CONVERT_ADD_EMBEDDINGS_URL')
        if process_all:
            url = f"{url}?process_all=true"
        try:
            response = requests.post(url)
            if response.status_code == 200:
                st.success(f"{response.text}\nPlease note this is an asynchronous process and may take a few minutes to complete.")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(traceback.format_exc())

    def delete_row():
        st.session_state['data_to_drop'] 
        redisembeddings.delete_document(st.session_state['data_to_drop'])

    def add_urls():
        urls = st.session_state['urls'].split('\n')
        for url in urls:
            if url:
                llm_helper.add_embeddings_lc(url)
                st.success(f"Embeddings added successfully for {url}")
    #needs a way to specify to translate or not, probably best to put it in params
    def post(self):
        try:
            res = {}
            llm_helper = LLMHelper()
            uploaded_file = request.files['file']
            if uploaded_file is not None:
                bytes_data = uploaded_file.read()
                content_type = mimetypes.MimeTypes().guess_type(uploaded_file.filename)[0]
                charset = f"; charset={chardet.detect(bytes_data)['encoding']}" if content_type == 'text/plain' else ''
                fileUrl = llm_helper.blob_client.upload_file(bytes_data, uploaded_file.filename, content_type=content_type+charset)
                converted_filename = ''
                if uploaded_file.filename.endswith('.txt'):
                    # Add the text to the embeddings
                    llm_helper.add_embeddings_lc(fileUrl)

                else:
                    # Get OCR with Layout API and then add embeddigns
                    converted_filename = llm_helper.convert_file_and_add_embeddings(fileUrl, uploaded_file.filename, False)
                
                llm_helper.blob_client.upsert_blob_metadata(uploaded_file.name, {'converted': 'true', 'embeddings_added': 'true', 'converted_filename': parse.quote(converted_filename)})
                return jsonify({f"File {uploaded_file.filename} embeddings added to the knowledge base."})
        except Exception as e:
            return jsonify({'error': f'{e}'})
        
class GetEmbeddingsModel(Resource):
    def get(self):
        try:
            res = LLMHelper().get_embeddings_model()
            return jsonify(res)
        except Exception as e:
            return jsonify({'error': f'{e}'})
        
class AddText(Resource):
    def post(self):
        try:
            data = request.get_json()
            llm_helper = LLMHelper()
            file_name = f"{uuid.uuid4()}.txt"
            sourceUrl = llm_helper.blob_client.upload_file(data["text"], file_name=file_name, content_type='text-plain; charset=utf-8')
            llm_helper.add_embeddings_lc(sourceUrl)
            return jsonify({'message': "Embeddings added successfully"})
        except Exception as e:
            return jsonify({'error': f'{e}'})
        
class AddBatch(Resource):
    def post(self):
        try:
            llm_helper = LLMHelper()
            uploaded_files = request.files
            if uploaded_files is not None:
                for up in uploaded_files:
                    bytes_data = up.read()
                    content_type = mimetypes.MimeTypes().guess_type(up.filename)[0]
                    charset = f"; charset={chardet.detect(bytes_data)['encoding']}" if content_type == 'text/plain' else ''
                    fileUrl = llm_helper.blob_client.upload_file(bytes_data, up.filename, content_type=content_type+charset)
        except Exception as e:
            return jsonify({'error': f'{e}'})
        
class ConvertAndAddEmbeddings(Resource):
    def post(self):
        try:
            data = request.get_json()
            url = os.getenv('CONVERT_ADD_EMBEDDINGS_URL')
            if data['process_all'] == True:
                url = f"{url}?process-all=true"
            response = requests.post(url)
            if response.status_code == 200:
                return jsonify({"message": f"{response.text}\nPlease note this is an asynchronous process and may take a few minutes to complete."})
            else:
                return jsonify({'Error': f"{response.text}"})
        except Exception as e:
            return jsonify({'error': f'{e}'})
        
class AddUrls(Resource):
    def post(self):
        res = {}
        try:
            llm_helper = LLMHelper()
            data = request.get_json()
            urls = data["urls"]
            for url in urls:
                if url:
                    llm_helper.add_embeddings_lc(url)
                    res[f"{url}"] = "success"
            return jsonify(res)
        except Exception as e:
            return jsonify({'error': f'{e}'})
        
class GetAllDocuments(Resource):
    def get(self):
        try:
            llm_helper = LLMHelper()
            data = llm_helper.get_all_documents(k=1000)
            if len(data) == 0:
                data = data.to_json()
                return jsonify({"data": data, "message":"No embeddings found."})
            else:
                data = data.to_json()
                return jsonify({"data": data})
        except Exception as e:
            if isinstance(e, ResponseError):
                return jsonify({"message": "No embeddings found."})
            else:
                return jsonify({'error': f'{e}'})
            


# adding the defined resources along with their corresponding urls 
api.add_resource(Hello, '/') 
api.add_resource(CheckDeployment, '/check-deployment')
api.add_resource(GetLanguages, '/get-languages')
api.add_resource(Query, '/query')
api.add_resource(Chat, '/chat')
api.add_resource(AddDocument, '/add-document')
api.add_resource(GetEmbeddingsModel, '/get-model')
api.add_resource(AddText, '/add-text')
api.add_resource(AddBatch, '/add-batch')
api.add_resource(ConvertAndAddEmbeddings, '/convert')
api.add_resource(AddUrls, '/add-urls')
api.add_resource(GetAllDocuments, '/get-documents')


# driver function 
if __name__ == '__main__': 

	app.run(debug = True) 
