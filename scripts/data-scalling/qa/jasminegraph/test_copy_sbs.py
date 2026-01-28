"""Copyright 2025 JasmineGraph Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import json
import logging
import os
import random
import socket
import subprocess
import sys
import time
import requests
logging.basicConfig(level=logging.INFO, format="%(message)s")
import shutil

# JasmineGraph master config
RESULT = subprocess.check_output(["hostname", "-I"]).decode().strip()
SERVER_IP = RESULT.split()[0]
# HOST = "127.0.0.1"
HOST = "10.8.100.248"

HDFS_PORT = "9000"
PORT = 7777
LINE_END = b"\r\n"

# Folder containing text files
TEXT_FOLDER = "gold_hipporag"

# LLM runner configuration
# LLM_RUNNERS = f"http://{SERVER_IP}:11450:1"
# LLM_RUNNERS = f"http://10.8.100.22:6578:4"
# REASONING_MODEL_URI = f"http://10.8.100.22:6578"
# LLM_RUNNERS = f"http://10.8.100.248:11450:4"
REASONING_MODEL_URI = f"http://10.8.100.248:11450"
LLM_RUNNERS = f"https://sajeenthiranp-21--h200-gpu-google-gemma-3-12b-it-vllm-serve.modal.run:4"
# REASONING_MODEL_URI = f"https://kumuthniparameswaran--h200-gpu-google-gemma-3-12b-it-vllm-serve.modal.run"
# LLM_MODEL = "google/gemma-3-4b-it"
# LLM_MODEL = "google/gemma-3-12b-it"
LLM_MODEL = "gemma3:12b"

# LLM_INFERENCE_ENGINE = "ollama"
LLM_INFERENCE_ENGINE = "vllm"

CHUNK_SIZE = "1500"

# HDFS target folder
HDFS_BASE = "/home/"

# Path to scripts
UPLOAD_SCRIPT = "../../utils/datasets/upload-hdfs-file.sh"
OLLAMA_SETUP_SCRIPT = "../utils/start-ollama.sh"


def recv_until(sock, stop=b"\n"):
    """Receive bytes from socket until a stop character."""
    buffer = bytearray()
    while True:
        chunk = sock.recv(1)
        if not chunk:
            break
        buffer.extend(chunk)
        if buffer.endswith(stop):
            break
    return buffer.decode("utf-8")


def upload_to_hdfs(local_file, upload_file_script):
    """Upload a local file to HDFS using the specified Bash script."""
    hdfs_filename = os.path.basename(local_file)
    hdfs_path = os.path.join(HDFS_BASE, hdfs_filename)

    logging.info("Uploading %s → HDFS:%s", local_file, hdfs_path)
    subprocess.run(["bash", upload_file_script, local_file, HDFS_BASE], check=True)
    return hdfs_path


def send_file_to_master(hdfs_file_path, host, port):
    """Send file path and configuration to JasmineGraph master."""
    logging.info("Sending %s to JasmineGraph master", hdfs_file_path)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        logging.info("Connected to JasmineGraph master at %s:%d", host, port)

        sock.sendall(b"constructkg" + LINE_END)
        msg = recv_until(sock, b"\n")
        logging.info("Master: %s", msg.strip())
        sock.sendall(b"n" + LINE_END)

        msg = recv_until(sock, b"\n")
        logging.info("Master: %s", msg.strip())
        sock.sendall(SERVER_IP.encode("utf-8") + LINE_END)

        msg = recv_until(sock, b"\n")
        logging.info("Master: %s", msg.strip())
        sock.sendall(HDFS_PORT.encode("utf-8") + LINE_END)

        msg = recv_until(sock, b"\n")
        logging.info("Master: %s", msg.strip())
        sock.sendall(hdfs_file_path.encode("utf-8") + LINE_END)

        msg = recv_until(sock, b"\n")
        logging.info("Master 101: %s", msg.strip())

        logging.info("LLM_RUNNERS: %s", LLM_RUNNERS)
        sock.sendall(LLM_RUNNERS.encode("utf-8") + LINE_END)

        msg = recv_until(sock, b"\n")
        logging.info("Master: %s", msg.strip())
        sock.sendall(LLM_INFERENCE_ENGINE.encode("utf-8") + LINE_END)

        msg = recv_until(sock, b"\n")
        logging.info("Master: %s", msg.strip())
        sock.sendall(LLM_MODEL.encode("utf-8") + LINE_END)

        msg = recv_until(sock, b"\n")
        logging.info("Master: %s", msg.strip())
        sock.sendall(CHUNK_SIZE.encode("utf-8") + LINE_END)

        final = recv_until(sock, b"\n").strip()

        if final == "There exists a graph with the file path, would you like to resume?":
            sock.sendall(b"n" + LINE_END)
            final = recv_until(sock, b"\n").strip()
            logging.info("Master: %s", final)
            sock.sendall(b"exit" + LINE_END)
            logging.info("KG extraction started successfully!")
        else:
            logging.info("Master: %s", final)
            sock.sendall(b"exit" + LINE_END)

        return final.split(":")[1] if ":" in final else final


def run_cypher_query(graph_id, query, host, port):
    """Run a Cypher query and return result rows."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        logging.info("Connected to JasmineGraph at %s:%d", host, port)

        sock.sendall(b"cypher" + LINE_END)
        recv_until(sock, b"\n")

        sock.sendall(graph_id.encode("utf-8") + LINE_END)
        recv_until(sock, b"\n")

        sock.sendall(query.encode("utf-8") + LINE_END)

        rows = []
        while True:
            line = recv_until(sock, b"\n").strip()
            if not line or "done" in line:
                break
            rows.append(line)

        sock.sendall(b"exit" + LINE_END)
        return rows


def parse_results(raw_rows):
    """Parse JSON rows from JasmineGraph into triples."""
    triples = []
    for row in raw_rows:
        try:
            data = json.loads(row)
            head = data["n"].get("name", data["n"].get("id"))
            tail = data["m"].get("name", data["m"].get("id"))
            rel = data["r"].get("type", "related_to")
            triples.append({"head_entity": head, "tail_entity": tail, "relation": rel})
        except (json.JSONDecodeError, KeyError) as err:
            logging.warning("Could not parse row: %s (%s)", row, err)
    return triples
def call_reasoning_model(prompt):
    """
    Call reasoning model (Ollama or vLLM) using the first runner URL.
    Uses LLM_INFERENCE_ENGINE ('ollama' | 'vllm') and LLM_MODEL.
    """
    if not REASONING_MODEL_URI:
        logging.error("❌ No valid runner URL found in LLM_RUNNERS")
        return ""

    try:
        if LLM_INFERENCE_ENGINE.lower() == "ollama":
            url = f"{REASONING_MODEL_URI}/api/generate"
            payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            print(data)
            return data.get("response", "").strip()

        elif LLM_INFERENCE_ENGINE.lower() == "vllm":
            url = f"{REASONING_MODEL_URI}/v1/chat/completions"
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a reasoning assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 512,
            }
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        else:
            raise ValueError(f"Unsupported engine: {LLM_INFERENCE_ENGINE}")

    except Exception as e:
        logging.error(f"❌ Reasoning model call failed: {e}")
        return ""
def compress_triples(triples):
    """
    Compress triples to short string representation:
    [{"head_entity":"A","relation":"is","tail_entity":"B"}] -> ["A|is|B"]
    """
    compressed = [
        f"{t['head_entity']}|{t['relation']}|{t['tail_entity']}" for t in triples
    ]
    return compressed
def run_sbs_query(graph_id, query, host, port):
    """Run SBS query and return JSON rows."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        logging.info("Connected to JasmineGraph at %s:%d for SBS", host, port)

        # Enter SBS mode
        sock.sendall(b"sbs" + LINE_END)
        recv_until(sock, b"\n")

        # Send graph ID
        sock.sendall(graph_id.encode("utf-8") + LINE_END)
        recv_until(sock, b"\n")

        # Send SBS natural-language query
        print("sbs query:", query)
        sock.sendall(query.encode("utf-8") + LINE_END)

        results = []
        while True:
            line = recv_until(sock, b"\n").strip()
            if not line or "done" in line:
                break
            results.append(line)

        sock.sendall(b"exit" + LINE_END)
        print(results)
        return results




def parse_sbs_results(raw_rows):
    """Parse SBS JSON lines into structured objects."""
    parsed = []
    s = set()
    for row in raw_rows:
        row_dict = json.loads(row)
        try:
            pathNodes = row_dict["pathObj"]["pathNodes"]
            pathRels = row_dict["pathObj"]["pathRels"]
            for idx, pathRel in enumerate(pathRels):
                if pathRel["id"] in s: continue
                s.add(pathRel["id"])
                if(pathRel["direction"]=="right"):
                    parsed.append(pathNodes[idx]["name"]+" "+ pathRel["type"]+" "+ pathNodes[idx+1]["name"])
                else:
                    parsed.append(pathNodes[idx+1]["name"]+" "+ pathRel["type"]+" "+ pathNodes[idx]["name"])


            # for rel in row_dict["pathObj"]["pathRels"]:
            #     if rel["id"] in s: continue
            #     s.add(rel["id"])
            #     parsed.append(rel["description"])
        except json.JSONDecodeError:
            logging.warning("Invalid SBS JSON row: %s", row)
    return parsed


# Sort files by numeric prefix of containing folder (e.g. `5_hipporag` -> 5);
# non-numeric folders come after numeric ones and are sorted lexicographically.
def folder_key(path):
    folder = os.path.basename(os.path.dirname(path))
    prefix = folder.split("_", 1)[0]
    try:
        num = int(prefix)
        return (0, num, folder, os.path.basename(path))
    except ValueError:
        return (1, folder, os.path.basename(path))

def test_kg(text_folder, upload_file_script, host, port):
    """Upload files, construct KGs, query them, and store triples."""
    query = "MATCH (n)-[r]-(m) RETURN n,r,m"
    all_txt_files = []

    for root, _, files in os.walk(text_folder):
        for file in files:
            if file.endswith(".txt"):
                all_txt_files.append(os.path.join(root, file))



    all_txt_files.sort(key=folder_key)
    all_txt_files = all_txt_files[0:]
    sbs_latencies = []
    # random.shuffle(all_txt_files)

    for local_path in all_txt_files:
        folder_name = os.path.basename(os.path.dirname(local_path))
        # try:
        #     # hdfs_path = upload_to_hdfs(local_path, upload_file_script)
        #     graph_id = send_file_to_master(hdfs_path, host, port)
        # except subprocess.CalledProcessError as err:
        #     logging.error("Failed to upload %s to HDFS: %s", local_path, err)
        #     continue

        # Wait for KG construction
        # time.sleep(60)
        graph_id = 13
        # raw = run_cypher_query(str(graph_id), query, host, port)
        # triples = parse_results(raw)
        # print(json.dumps(triples, indent=2, ensure_ascii=False))
        output_dir = os.path.join("pred", folder_name)
        os.makedirs(output_dir, exist_ok=True)
        #
        # with open(os.path.join(output_dir, "pred.json"), "w", encoding="utf-8") as f:
        #     json.dump(triples, f, indent=2, ensure_ascii=False)

        # ------------ SBS QUERY HANDLING ------------
        qa_file = os.path.join(text_folder, folder_name, "qa_pairs.json")

        if os.path.exists(qa_file):
            with open(qa_file, "r", encoding="utf-8") as f:
                qa_data = json.load(f)

            question = qa_data["question"]
            answer = qa_data["gold_answer"] if "gold_answer" in qa_data else qa_data["answer"]

            # --- Run SBS query ---
            start_time = time.time()   # record start
            sbs_raw = run_sbs_query(str(graph_id), question, host, port)
            end_time = time.time()     # record end

            sbs_latency = end_time - start_time
            logging.info("✅ SBS query latency: %.3f seconds", sbs_latency)
            sbs_latencies.append(sbs_latency)

            sbs_parsed = parse_sbs_results(sbs_raw)
            print(sbs_parsed)
            # Save SBS results

            print(json.dumps(sbs_parsed, indent=2, ensure_ascii=False))
            output_dir = os.path.join("pred", folder_name)
            os.makedirs(output_dir, exist_ok=True)

            with open(os.path.join(output_dir, "sbs.json"), "w", encoding="utf-8") as f:
                json.dump(sbs_parsed, f, indent=2, ensure_ascii=False)

            # sbs_out_path = os.path.join(output_dir, "sbs.json")
            # with open(sbs_out_path, "w", encoding="utf-8") as sf:
            #     json.dump(sbs_parsed,   sf, indent=2, ensure_ascii=False)
            # Copy gold files
        for gold_file in ["text.txt"]:
            src = os.path.join(text_folder, folder_name, gold_file)
            dst = os.path.join(output_dir, gold_file)
            if os.path.exists(src):
                shutil.copy(local_path, dst)

        # QA prediction
        qa_file = os.path.join(text_folder, folder_name, "qa_pairs.json")
        if os.path.exists(qa_file):
            print("file exists")
            with open(qa_file, "r", encoding="utf-8") as f:
                qa_data = json.load(f)

            question = qa_data["question"]
            answer = qa_data["answer"]
            # compressed_triples_ = compress_triples(triples)


            # Load SBS results if available
            sbs_path = os.path.join(output_dir, "sbs.json")
            if os.path.exists(sbs_path):
                with open(sbs_path, "r", encoding="utf-8") as sf:
                    sbs_results = json.load(sf)
            else:
                sbs_results = []

            prompt = f"""You are a QA assistant.
Given the following context:
{json.dumps(sbs_results, indent=2, ensure_ascii=False)}

Question: {question}
 Answer in a short phrase. Do not include reasoning, explanations, or extra text."""

            predicted_answer = call_reasoning_model(prompt)

            pred_out = {
                "id": qa_data["id"],
                "question": question,
                "gold_answer": answer,
                "predicted_answer": predicted_answer,
            }
            with open(
                    os.path.join(output_dir, "pred_answer.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(pred_out, f, indent=2, ensure_ascii=False)
        # if not triples or triples[0].get("head_entity") != "Radio City":
        #     logging.error("Expected first head_entity to be 'Radio City', got: %s",
        #                   triples[0] if triples else None)
        #     sys.exit(1)
        #
        # if not triples or triples[1].get("head_entity") != "Radio City":
        #     logging.error("Expected first head_entity to be 'Radio City', got: %s",
        #                   triples[0] if triples else None)
        #     sys.exit(1)
        #
        # if not triples or triples[2].get("head_entity") != "Radio City":
        #     logging.error("Expected first head_entity to be 'Radio City', got: %s",
        #                   triples[0] if triples else None)
        #     sys.exit(1)
        # raw = run_cypher_query(str(graph_id), query, host, port)
        # triples = parse_results(raw)
        # print(json.dumps(triples, indent=2, ensure_ascii=False))
        # output_dir = os.path.join("pred", folder_name)
        # os.makedirs(output_dir, exist_ok=True)
        #
        # with open(os.path.join(output_dir, "pred.json"), "w", encoding="utf-8") as f:
        #     json.dump(triples, f, indent=2, ensure_ascii=False)
        #
        # output_dir = os.path.join("pred", folder_name)
        # os.makedirs(output_dir, exist_ok=True)
        #
        # with open(os.path.join(output_dir, "pred.json"), "w", encoding="utf-8") as f:
        #     json.dump(triples, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    folder = TEXT_FOLDER
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        print("testing folder:", folder)
    test_kg(folder, UPLOAD_SCRIPT, HOST, PORT)
