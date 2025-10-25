import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import shutil
import glob

app = FastAPI()

# ✅ IMPORTANT: Serve static files so frontend can fetch BVH
PUBLIC_DIR = "../web/mesh/public"
os.makedirs(PUBLIC_DIR, exist_ok=True)
app.mount("/mesh/public", StaticFiles(directory=PUBLIC_DIR), name="mesh")

# ✅ NEW: Serve JSON motion data from frontend's public directory
JSON_PUBLIC_DIR = "../web/public/motionjson"
os.makedirs(JSON_PUBLIC_DIR, exist_ok=True)
app.mount("/motionjson", StaticFiles(directory=JSON_PUBLIC_DIR), name="motionjson")

# ✅ Enable CORS for ngrok
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚀 Global model instance - loaded once at startup
generator = None

@app.on_event("startup")
def load_models():
    """Load models once at startup to avoid reloading on every request"""
    global generator
    print("🚀 Loading MaskControl models at startup...")
    from generation.maskcontrol_generator import MaskControlGenerator
    generator = MaskControlGenerator(device="cuda:0")
    print("✅ Models loaded and ready!")

from pydantic import BaseModel
from typing import List, Optional

class Point(BaseModel):
    x: float
    y: float

class TrajectoryRequest(BaseModel):
    path: List[Point]
    canvas_width: float
    canvas_height: float
    text_prompt: Optional[str] = "a person walks forward"  # ✅ Add text prompt support


@app.post("/maskcontrol/set-trajectory")
def set_trajectory(req: TrajectoryRequest):
    try:
        import numpy as np
        
        pts = np.array([[p.x, p.y] for p in req.path], dtype=np.float32)
        if len(pts) < 2:
            raise HTTPException(status_code=400, detail="Path too short")
        
        print(f"📍 Received {len(pts)} canvas points")
        print(f"   First: canvas X={pts[0,0]:.1f}, Y={pts[0,1]:.1f}")
        print(f"   Last:  canvas X={pts[-1,0]:.1f}, Y={pts[-1,1]:.1f}")
        
        # Map canvas coordinates to model space
        # Canvas X (0-280) → Model X coordinate (-1 to 1)
        # Canvas Y (0-200) → Model Z coordinate (-1 to 1), inverted
        
        model_x = (pts[:, 0] - req.canvas_width / 2) / (req.canvas_width / 2)
        model_z = -(pts[:, 1] - req.canvas_height / 2) / (req.canvas_height / 2)
        model_y = np.full_like(model_x, 0.8, dtype=np.float32)
        
        # Scale to match circle trajectory range (circle has radius=1, spans 2 units)
        model_x = -model_x * 2.0  # Negate and scale to match default trajectory magnitude
        model_z = model_z * 2.0    # Scale to match default trajectory magnitude
        
        # Calculate cumulative distance along the path for constant speed resampling
        points_3d = np.column_stack([model_x, model_y, model_z])
        distances = np.sqrt(np.sum(np.diff(points_3d, axis=0)**2, axis=1))
        cumulative_dist = np.concatenate([[0], np.cumsum(distances)])
        total_distance = cumulative_dist[-1]
        
        # Model requires fixed frame count of 196
        num_frames = 196
        
        # Determine target speed (units per frame) - match straight line walking
        target_speed = 0.01  # units per frame (same as draw_straight_line)
        natural_frames = int(total_distance / target_speed)
        actual_speed = total_distance / num_frames
        
        print(f"📏 Path length: {total_distance:.2f} units")
        print(f"🏃 Natural frames for walking speed: {natural_frames}")
        print(f"⏱️  Fixed at 196 frames (model requirement)")
        print(f"⏱️  Actual speed: {actual_speed:.4f} units/frame (target: {target_speed:.4f})")
        
        if abs(actual_speed - target_speed) > 0.005:
            print(f"⚠️  Speed differs from natural walking - motion may look fast/slow")
        
        # Resample at even distances along the path (constant speed)
        target_distances = np.linspace(0, total_distance, num_frames)
        model_x_resampled = np.interp(target_distances, cumulative_dist, model_x)
        model_z_resampled = np.interp(target_distances, cumulative_dist, model_z)
        
        # Add natural vertical hip oscillation (matches draw_circle_with_waves)
        wave_amplitude = 0.1
        wave_frequency = 5
        angles = np.linspace(0, 2 * np.pi * wave_frequency, num_frames)
        model_y_resampled = 0.8 + wave_amplitude * np.sin(angles)
        model_y_resampled = model_y_resampled.astype(np.float32)
        
        print(f"🌊 Added hip oscillation: Y range [{model_y_resampled.min():.2f}, {model_y_resampled.max():.2f}]")
        
        # ✅ CRITICAL: Match draw_circle_with_waves format: [-X, Y, Z]
        traj = np.stack([model_x_resampled, model_y_resampled, model_z_resampled], axis=1).astype(np.float32)
        
        np.save("frontend_traj.npy", traj)
        
        # ✅ Save text prompt alongside trajectory
        if req.text_prompt:
            with open("frontend_traj_prompt.txt", "w") as f:
                f.write(req.text_prompt)
            print(f"✅ Saved text prompt: '{req.text_prompt}'")
        
        print(f"✅ Saved trajectory in [-X, Y, Z] format (matches draw_circle_with_waves):")
        print(f"   Shape: {traj.shape}")
        print(f"   First frame: [{traj[0,0]:.3f}, {traj[0,1]:.3f}, {traj[0,2]:.3f}]")
        print(f"   Last frame:  [{traj[-1,0]:.3f}, {traj[-1,1]:.3f}, {traj[-1,2]:.3f}]")
        print(f"   X range: [{traj[:,0].min():.3f}, {traj[:,0].max():.3f}]")
        print(f"   Z range: [{traj[:,2].min():.3f}, {traj[:,2].max():.3f}]")
        
        return {"success": True, "message": "Trajectory and prompt saved"}
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

class AvoidanceRequest(BaseModel):
    text_prompt: str = "a person walks forward"
    motion_length: int = 196
    control_frame: int = 195
    control_joint: int = 0
    iter_each: int = 0
    iter_last: int = 100
    avoidance_points: Optional[List[List[float]]] = None

@app.post("/maskcontrol/generate-avoidance")
def generate_avoidance(req: AvoidanceRequest = None):
    """Generate motion with avoidance using MaskControlGenerator"""
    try:
        # Use default if no request body provided
        if req is None:
            req = AvoidanceRequest()
        
        # Use the global generator instance (loaded at startup)
        global generator
        if generator is None:
            raise HTTPException(status_code=503, detail="Models not loaded yet")
        
        output_dir = "./output/avoidance_web2"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"✓ Starting MaskControl avoidance generation...")
        print(f"  Text: {req.text_prompt}")
        print(f"  Iterations: each={req.iter_each}, last={req.iter_last}")
        
        # Process avoidance points from frontend
        avoid_points_tensor = None
        if req.avoidance_points is not None and len(req.avoidance_points) > 0:
            import torch
            print(f"✓ Received {len(req.avoidance_points)} avoidance points from frontend:")
            for i, point in enumerate(req.avoidance_points):
                x, y, z, r = point[0], point[1], point[2], point[3]
                print(f"  Point {i}: X={x:.4f}, Y={y:.4f}, Z={z:.4f}, R={r:.4f}")
            avoid_points_tensor = torch.tensor(req.avoidance_points, dtype=torch.float32, device='cuda')
        else:
            print(f"⚠ No avoidance points provided, using default obstacles")
        
        # Generate motion using the preloaded generator
        bvh_path = generator.generate_with_avoidance(
            text_prompt=req.text_prompt,
            motion_length=req.motion_length,
            control_frame=req.control_frame,
            control_joint=req.control_joint,
            control_pos=(0.5, 1.0229, 6.5),
            avoid_points=avoid_points_tensor,  # Pass frontend points or None for defaults
            out_dir=output_dir,
            out_name="avoidance.bvh",
            iter_each=req.iter_each,
            iter_last=req.iter_last,
            export_json=True  # ✅ Also export JSON for frontend retargeting
        )
        
        print(f"✓ Generated BVH: {bvh_path}")
        
        # Copy BVH to public directory
        bvh_filename = "avoidance_output.bvh"
        bvh_final = os.path.join(PUBLIC_DIR, bvh_filename)
        shutil.copyfile(bvh_path, bvh_final)
        print(f"✓ BVH accessible at: /mesh/public/{bvh_filename}")
        
        # Copy JSON to new motionjson directory (if exists)
        json_filename = "avoidance_output.json"
        json_source = bvh_path.replace('.bvh', '.json')
        json_final = os.path.join(JSON_PUBLIC_DIR, json_filename)
        motion_data = None
        
        if os.path.exists(json_source):
            shutil.copyfile(json_source, json_final)
            print(f"✓ JSON accessible at: /motionjson/{json_filename}")
            
            # Load JSON data to send in response
            import json
            with open(json_source, 'r') as f:
                motion_data = json.load(f)
            print(f"✓ Loaded motion data: {motion_data['num_frames']} frames")
        
        return {
            "success": True,
            "filename": bvh_filename,  # ✅ Backward compatible field
            "bvh_filename": bvh_filename,
            "json_filename": json_filename,
            "json_url": f"/motionjson/{json_filename}",
            "motion_data": motion_data,  # ✅ Send JSON data in response
            "message": "Generated motion with JSON support"
        }
        
    except Exception as e:
        import traceback
        print(f"✗ Full error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


class ControlJointRequest(BaseModel):
    text_prompt: str = "a person walks in a circle counter-clockwise"
    motion_length: int = 196
    iter_each: int = 100  # Increased from 0 - optimize at each unmask step for smoother motion
    iter_last: int = 600  # Increased from 100 - more final optimization for better constraint satisfaction
    path: List[Point] = None  # Optional path points
    canvas_width: float = None
    canvas_height: float = None


@app.post("/maskcontrol/generate-custom")
def generate_custom(req: ControlJointRequest):
    """Generate motion with controlled joint trajectories using MaskControlGenerator"""
    try:
        import numpy as np
        import torch
        
        # Use the global generator instance (loaded at startup)
        global generator
        if generator is None:
            raise HTTPException(status_code=503, detail="Models not loaded yet")
        
        output_dir = "./output/control_joint_web"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"✓ Starting control_joint generation...")
        
        # ✅ Load saved text prompt if available and not overridden
        text_prompt = req.text_prompt
        if os.path.exists("./frontend_traj_prompt.txt") and req.text_prompt == "a person walks in a circle counter-clockwise":
            with open("./frontend_traj_prompt.txt", "r") as f:
                text_prompt = f.read().strip()
            print(f"✓ Loaded saved text prompt: '{text_prompt}'")
        
        print(f"  Text: {text_prompt}")
        print(f"  Length: {req.motion_length}")
        
        traj1 = None
        
        # Priority 1: Use path points if provided in request
        if req.path and req.canvas_width and req.canvas_height:
            print(f"✓ Using path from request ({len(req.path)} points)")
            
            # Convert list of Points → numpy (N, 2)
            pts = np.array([[p.x, p.y] for p in req.path], dtype=np.float32)
            
            if len(pts) < 2:
                raise HTTPException(status_code=400, detail="Path too short")
            
            # Center & normalize to [-1..1]
            pts[:, 0] = (pts[:, 0] - req.canvas_width / 2) / (req.canvas_width / 2)
            pts[:, 1] = -(pts[:, 1] - req.canvas_height / 2) / (req.canvas_height / 2)
            
            # Resample to motion_length frames
            idx = np.linspace(0, len(pts)-1, req.motion_length).astype(int)
            traj = np.zeros((req.motion_length, 3), dtype=np.float32)
            traj[:, 0:2] = pts[idx]  # XY, Z stays 0
            
            traj1 = torch.tensor(traj).float().cuda()
            
        # Priority 2: Check if custom trajectory exists from previous /set-trajectory call
        elif os.path.exists("./frontend_traj.npy"):
            print(f"✓ Loading custom trajectory from: ./frontend_traj.npy")
            traj1 = torch.tensor(np.load("./frontend_traj.npy")).float().cuda()
            # Trajectory is always 196 frames (model requirement)
            
        # Priority 3: Use default circular trajectory (will be set by generator)
        else:
            print(f"✓ Using default circular trajectory")
        
        # Generate motion (optionally with JSON)
        print(f"🚀 Calling generator with export_json=True and export_html=True")
        bvh_path = generator.generate_with_control_joint(
            text_prompt=text_prompt,  # ✅ Use loaded/custom prompt
            motion_length=req.motion_length,
            traj1=traj1,  # Will use default if None
            traj2=None,   # Use default for joint 20
            out_dir=output_dir,
            out_name="control_joint.bvh",
            iter_each=req.iter_each,
            iter_last=req.iter_last,
            export_json=True,  # ✅ Also export JSON for frontend retargeting
            export_html=True   # ✅ Also export HTML skeleton visualization
        )
        
        print(f"✓ Generated BVH: {bvh_path}")
        print(f"✓ Generator returned successfully")
        
        # Copy BVH to public directory
        bvh_filename = "control_joint_output.bvh"
        bvh_final = os.path.join(PUBLIC_DIR, bvh_filename)
        shutil.copyfile(bvh_path, bvh_final)
        print(f"✓ BVH accessible at: /mesh/public/{bvh_filename}")
        
        # Copy JSON to new motionjson directory (if exists)
        json_filename = "control_joint_output.json"
        json_source = bvh_path.replace('.bvh', '.json')
        json_final = os.path.join(JSON_PUBLIC_DIR, json_filename)
        motion_data = None
        
        print(f"🔍 Looking for JSON at: {json_source}")
        print(f"🔍 JSON exists: {os.path.exists(json_source)}")
        
        if os.path.exists(json_source):
            print(f"✅ Found JSON file, copying to: {json_final}")
            shutil.copyfile(json_source, json_final)
            print(f"✓ JSON accessible at: /motionjson/{json_filename}")
            
            # Load JSON data to send in response
            import json
            with open(json_source, 'r') as f:
                motion_data = json.load(f)
            print(f"✓ Loaded motion data: {motion_data['num_frames']} frames")
        else:
            print(f"❌ JSON file NOT FOUND at: {json_source}")
            print(f"❌ This means export_joints_json() was not called or failed")
        
        return {
            "success": True,
            "filename": bvh_filename,  # ✅ Backward compatible field
            "bvh_filename": bvh_filename,
            "json_filename": json_filename,
            "json_url": f"/motionjson/{json_filename}",
            "motion_data": motion_data,  # ✅ Send JSON data in response
            "message": "Generated motion with JSON support"
        }
        
    except Exception as e:
        import traceback
        print(f"✗ Full error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/maskcontrol/generate-control-joint")
def generate_control_joint_default():
    """Generate motion with default control_joint settings using MaskControlGenerator"""
    try:
        # Use the global generator instance (loaded at startup)
        global generator
        if generator is None:
            raise HTTPException(status_code=503, detail="Models not loaded yet")
        
        output_dir = "./output/control22222"
        
        # 🔧 FULL CLEAN: Remove entire output_dir if it exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)   # Removes everything including /source
        os.makedirs(output_dir, exist_ok=True)

        print("✓ Running default control_joint with preloaded generator...")

        # Generate motion using the preloaded generator with default settings
        print(f"🚀 Calling generator with export_json=True")
        bvh_path = generator.generate_with_control_joint(
            text_prompt="a person walks in a circle counter-clockwise",
            motion_length=196,
            traj1=None,  # Use default circular trajectory
            traj2=None,  # Use default for joint 20
            out_dir=output_dir,
            out_name="generation.bvh",
            iter_each=0,
            iter_last=100,
            export_json=True  # ✅ Enable JSON export
        )

        print(f"✓ Generated BVH: {bvh_path}")

        # Copy BVH to public
        bvh_filename = "control_joint_output.bvh"
        bvh_final = os.path.join(PUBLIC_DIR, bvh_filename)
        shutil.copyfile(bvh_path, bvh_final)
        print(f"✓ BVH ready at /mesh/public/{bvh_filename}")
        
        # Copy JSON to new motionjson directory (if exists)
        json_filename = "control_joint_output.json"
        json_source = bvh_path.replace('.bvh', '.json')
        json_final = os.path.join(JSON_PUBLIC_DIR, json_filename)
        motion_data = None
        
        print(f"🔍 Looking for JSON at: {json_source}")
        print(f"🔍 JSON exists: {os.path.exists(json_source)}")
        
        if os.path.exists(json_source):
            print(f"✅ Found JSON file, copying to: {json_final}")
            shutil.copyfile(json_source, json_final)
            print(f"✓ JSON accessible at: /motionjson/{json_filename}")
            
            # Load JSON data to send in response
            import json
            with open(json_source, 'r') as f:
                motion_data = json.load(f)
            print(f"✓ Loaded motion data: {motion_data['num_frames']} frames")
        else:
            print(f"❌ JSON file NOT FOUND at: {json_source}")
        
        return {
            "success": True,
            "filename": bvh_filename,  # ✅ Backward compatible field
            "bvh_filename": bvh_filename,
            "json_filename": json_filename,
            "json_url": f"/motionjson/{json_filename}",
            "motion_data": motion_data,
            "message": "Generated motion with JSON support"
        }

    except Exception as e:
        import traceback
        print("✗ Full error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)