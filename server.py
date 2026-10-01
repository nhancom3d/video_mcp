import os
import subprocess
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from mcp.server.fastmcp import FastMCP

MODEL_ID = os.getenv("MODEL_ID", "suzii/vi-whisper-large-v3-turbo")
DATA_DIR = "/app/data"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 1. Nạp Model
print(f"Loading {MODEL_ID} on {device}...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(MODEL_ID)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True
)

def format_timestamp(seconds: float) -> str:
    millis = int((seconds - int(seconds)) * 1000)
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"

# 2. Logic lõi xử lý Subtitle & Frame
def core_generate_sub(video_path: str, output_srt: str) -> str:
    result = pipe(video_path, generate_kwargs={"language": "vi", "task": "transcribe"})
    srt_lines = []
    for idx, chunk in enumerate(result.get("chunks", []), start=1):
        start, end = chunk["timestamp"]
        end = end or (start + 2.0)
        srt_lines.append(f"{idx}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{chunk['text'].strip()}\n")
    
    with open(output_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    return output_srt

def core_extract_frame(video_path: str, timestamp: str, output_img: str) -> str:
    cmd = ["ffmpeg", "-y", "-ss", timestamp, "-i", video_path, "-frames:v", "1", "-q:v", "2", output_img]
    subprocess.run(cmd, check=True)
    return output_img

# 3. Tạo FastMCP Server
mcp = FastMCP("Vietnamese-Video-Tools")

@mcp.tool()
def extract_frame_at_time(video_filename: str, timestamp: str) -> str:
    """Chụp ảnh từ video tại mốc thời gian (VD: '00:01:50' hoặc '110')"""
    v_path = os.path.join(DATA_DIR, video_filename)
    out_name = f"frame_{timestamp.replace(':', '_')}.jpg"
    out_path = os.path.join(DATA_DIR, out_name)
    core_extract_frame(v_path, timestamp, out_path)
    return f"Frame saved at: {out_path}"

@mcp.tool()
def generate_vietnamese_subtitles(video_filename: str) -> str:
    """Tạo phụ đề tiếng Việt SRT có timestamp đầy đủ từ video"""
    v_path = os.path.join(DATA_DIR, video_filename)
    out_srt = os.path.join(DATA_DIR, f"{os.path.splitext(video_filename)[0]}.srt")
    core_generate_sub(v_path, out_srt)
    return f"SRT Subtitle generated at: {out_srt}"

# 4. Tạo FastAPI và mount MCP Sub-Application
api = FastAPI(title="Media AI & MCP Gateway")

# Mount MCP SSE app vào đường dẫn /mcp
api.mount("/mcp", mcp.sse_app())

# Các REST endpoint thông thường
@api.post("/api/subtitles")
async def api_subtitles(file: UploadFile = File(...)):
    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    srt_path = os.path.join(DATA_DIR, f"{os.path.splitext(file.filename)[0]}.srt")
    core_generate_sub(save_path, srt_path)
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"status": "success", "srt_content": content}

@api.post("/api/extract-frame")
async def api_frame(file: UploadFile = File(...), timestamp: str = Form("00:01:50")):
    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    out_name = f"frame_{timestamp.replace(':', '_')}.jpg"
    out_path = os.path.join(DATA_DIR, out_name)
    core_extract_frame(save_path, timestamp, out_path)
    return {"status": "success", "image_path": out_path}

if __name__ == "__main__":
    # Chỉ cần chạy duy nhất một tiến trình Uvicorn trên 1 port
    uvicorn.run(api, host="0.0.0.0", port=8000)