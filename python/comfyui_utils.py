import websocket
from pathlib import Path
import numpy as np
import base64
import requests
import json
import dill
import urllib
from PIL import Image
from deepface import DeepFace
from PIL import Image
from pathlib import Path
from cog_server_utils import get_uuid, b2_upload_file_and_get_url
# ref: https://9elements.com/blog/hosting-a-comfyui-workflow-via-api/


# comfyui functions:

def open_websocket_connection(client_id=None):
  server_address='127.0.0.1:8081'
  client_id=get_uuid() if client_id is None else client_id
  ws = websocket.WebSocket()
  ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
  return ws, server_address, client_id


def queue_prompt(prompt, client_id, server_address):
  p = {"prompt": prompt, "client_id": client_id}
  headers = {'Content-Type': 'application/json'}
  data = json.dumps(p).encode('utf-8')
  req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data, headers=headers)
  return json.loads(urllib.request.urlopen(req).read())


def get_history(prompt_id, server_address):
  with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
      return json.loads(response.read())

def track_progress(prompt, ws, prompt_id, verbose=False):
  node_ids = list(prompt.keys())
  finished_nodes = []

  while True:
      out = ws.recv()
      if isinstance(out, str):
          message = json.loads(out)
          if message['type'] == 'progress':
              data = message['data']
              current_step = data['value']
              if verbose:
                print('In K-Sampler -> Step: ', current_step, ' of: ', data['max'])
          if message['type'] == 'execution_cached':
              data = message['data']
              for itm in data['nodes']:
                  if itm not in finished_nodes:
                      finished_nodes.append(itm)
                      if verbose:
                        print('Progess: ', len(finished_nodes), '/', len(node_ids), ' Tasks done')
          if message['type'] == 'executing':
              data = message['data']
              if data['node'] not in finished_nodes:
                  finished_nodes.append(data['node'])
                  if verbose:
                    print('Progess: ', len(finished_nodes), '/', len(node_ids), ' Tasks done')

              if data['node'] is None and data['prompt_id'] == prompt_id:
                  break #Execution is done
      else:
          continue
  return

face_extraction_workflow_path = Path.cwd() / 'comfyui-workflows' / 'extract-face-utl-workflow.json'
face_extraction_no_driving_workflow_path = Path.cwd() / 'comfyui-workflows' / 'extract-face-utl-no-driving-workflow.json'

def comfyui_generate_face_expression(input_data):
    ws = None

    try:
        ws, server_address, client_id = open_websocket_connection()

        print('setting submitExpressionEditorToFace:', face_extraction_workflow_path, 'with input:', input_data)

        if 'driving_image' in input_data and input_data['driving_image'] is not None and input_data['driving_image'] != '':
            with face_extraction_workflow_path.open('r') as file:
                local_workflow = json.load(file)
        else:
            with face_extraction_no_driving_workflow_path.open('r') as file:
                local_workflow = json.load(file)

        local_workflow['35']['inputs']['url_or_path'] = input_data['source_image']

        if 'rotate_pitch' in input_data:
            local_workflow['21']['inputs'].update({
                'rotate_pitch': input_data['rotate_pitch'] if 'rotate_pitch' in input_data else local_workflow['21']['inputs']['rotate_pitch'],
                'rotate_yaw': input_data['rotate_yaw'] if 'rotate_yaw' in input_data else local_workflow['21']['inputs']['rotate_yaw'],
                'rotate_roll': input_data['rotate_roll'] if 'rotate_roll' in input_data else local_workflow['21']['inputs']['rotate_roll'],
                'blink': input_data['blink'] if 'blink' in input_data else local_workflow['21']['inputs']['blink'],
                'eyebrow': input_data['eyebrow'] if 'eyebrow' in input_data else local_workflow['21']['inputs']['eyebrow'],
                'wink': input_data['wink'] if 'wink' in input_data else local_workflow['21']['inputs']['wink'],
                'pupil_x': input_data['pupil_x'] if 'pupil_x' in input_data else local_workflow['21']['inputs']['pupil_x'],
                'pupil_y': input_data['pupil_y'] if 'pupil_y' in input_data else local_workflow['21']['inputs']['pupil_y'],
                'aaa': input_data['aaa'] if 'aaa' in input_data else local_workflow['21']['inputs']['aaa'],
                'eee': input_data['eee'] if 'eee' in input_data else local_workflow['21']['inputs']['eee'],
                'woo': input_data['woo'] if 'woo' in input_data else local_workflow['21']['inputs']['woo'],
                'smile': input_data['smile'] if 'smile' in input_data else local_workflow['21']['inputs']['smile'],
                'src_ratio': input_data['src_ratio'] if 'src_ratio' in input_data else local_workflow['21']['inputs']['src_ratio'],
                'sample_ratio': input_data['sample_ratio'] if 'sample_ratio' in input_data else local_workflow['21']['inputs']['sample_ratio'],
                'sample_parts': input_data['sample_parts'] if 'sample_parts' in input_data else local_workflow['21']['inputs']['sample_parts'],
                'crop_factor': input_data['crop_factor'] if 'crop_factor' in input_data else local_workflow['21']['inputs']['crop_factor'],
                'sample_parts': input_data['sample_parts'] if 'sample_parts' in input_data else local_workflow['21']['inputs']['sample_parts'],
            })

        if 'driving_image' in input_data and input_data['driving_image'] is not None and input_data['driving_image'] != '':
            local_workflow['36']['inputs']['url_or_path'] = input_data['driving_image']

        TEST_DRIVING_IMAGE = False
        if TEST_DRIVING_IMAGE:
            local_workflow['36']['inputs']['url_or_path'] = 'https://static.crazyfaceai.com/b2api/v2/b2_download_file_by_id?fileId=4_zfb60fe55442959ea87f60d1e_f1188c14badecb4cc_d20240827_m214742_c005_v0501024_t0050_u01724795262743'

        print('submitting local_workflow:', local_workflow)

        result = queue_prompt(local_workflow, client_id, server_address)
        print('result:', result)
        if 'prompt_id' not in result:
            raise Exception('No prompt_id found in the result')

        prompt_id = result['prompt_id']
        print(f"queued: {prompt_id}")

        # track progress
        track_progress(local_workflow, ws, prompt_id)

        # get output:
        history = get_history(prompt_id, server_address)[prompt_id]

        print(f"history: {history}")

        output_path = None

        for image in history['outputs']['33']['images']:

            if image['type'] == 'output':
                # print((image['filename'], image['subfolder'], image['type']))
                output_path = Path.home() / "ComfyUI" / "output" / image['filename']
                print(output_path)
                break
            
        if output_path is None:
            raise Exception('No output path found')
        # upload to b2
        b2_url = b2_upload_file_and_get_url(file_path=output_path)

        to_ret =  {'id': prompt_id, 'output': b2_url}

        if False and 'driving_image' in input_data:
            exp_data_path = Path.home() / "ComfyUI" / "output" / "exp_data" / "ExpressionData.exp"
            if exp_data_path.exists():
                # load dill data
                with exp_data_path.open('rb') as file:
                    to_ret['exp_data'] = dill.load(file)
                    print(f"exp_data: {to_ret['exp_data']}")

        return to_ret

    except Exception as error:
        print(f"error: {error}")
        return {"error": str(error)}
    finally:
        if ws:
            ws.close()

