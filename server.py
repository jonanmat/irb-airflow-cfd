from __future__ import annotations

import math
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
JOBS_DIR = Path(os.getenv("IRB_JOBS_DIR", "/tmp/irb-cfd-jobs"))
FOAM_BASHRC = os.getenv("FOAM_BASHRC", "/usr/lib/openfoam/openfoam2312/etc/bashrc")
MAX_RUNTIME_SECONDS = int(os.getenv("IRB_MAX_RUNTIME_SECONDS", "1800"))
MAX_LOG_CHARS = 30000
MAX_PLANE_POINTS = 4200
MAX_VOLUME_POINTS = int(os.getenv("IRB_MAX_VOLUME_POINTS", "9000"))
MAX_ESTIMATED_CELLS = int(os.getenv("IRB_MAX_ESTIMATED_CELLS", "350000"))
DX = float(os.getenv("IRB_MESH_DX", "0.15"))
DY = float(os.getenv("IRB_MESH_DY", "0.10"))
DZ = float(os.getenv("IRB_MESH_DZ", "0.10"))
API_VERSION = "0.20.0"
BUILD = "IRB-CFD-OPENFOAM-0.20"

app = FastAPI(title="IRB AirFlow CFD Online", version=API_VERSION)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

jobs: Dict[str, dict] = {}
jobs_lock = threading.Lock()
active_processes: Dict[str, subprocess.Popen] = {}
solver_gate = threading.Semaphore(int(os.getenv("IRB_MAX_CONCURRENT_JOBS", "1")))

Wall = Literal["x0", "xL", "y0", "yW", "z0", "zH"]
Kind = Literal["supply", "return"]

class Room(BaseModel):
    L: float = Field(gt=0.2, le=50)
    W: float = Field(gt=0.2, le=50)
    H: float = Field(gt=0.2, le=15)

class Terminal(BaseModel):
    id: str = Field(min_length=1, max_length=50)
    name: str = Field(default="Terminal", min_length=1, max_length=80)
    kind: Kind
    wall: Wall
    a: float = Field(ge=0)
    b: float = Field(ge=0)
    width: float = Field(gt=0.03, le=10)
    height: float = Field(gt=0.03, le=10)
    q: float = Field(default=0, ge=0, le=100000)
    speed: Optional[float] = Field(default=None, gt=0, le=30)
    dir_x: Optional[float] = None
    dir_y: Optional[float] = None
    dir_z: Optional[float] = None
    supply_temp: Optional[float] = Field(default=None, ge=-30, le=80)

    @model_validator(mode="after")
    def supply_requires_flow(self):
        if self.kind == "supply" and self.q <= 0 and not self.speed:
            raise ValueError("Una impulsión necesita caudal o velocidad.")
        return self

class CFDRequest(BaseModel):
    room: Room
    terminals: List[Terminal] = Field(min_length=2, max_length=30)
    ambient_temp: float = Field(default=22, ge=-20, le=60)

class JobRequest(BaseModel):
    project: CFDRequest

class CancelRequest(BaseModel):
    id: str

def wall_axes(wall: Wall):
    if wall in ("x0", "xL"):
        return 0, 1, 2
    if wall in ("y0", "yW"):
        return 1, 0, 2
    return 2, 0, 1

def axis_length(room: Room, axis: int) -> float:
    return (room.L, room.W, room.H)[axis]

def terminal_limits(t: Terminal):
    return t.a - t.width / 2, t.a + t.width / 2, t.b - t.height / 2, t.b + t.height / 2

def normal_inward(wall: Wall) -> tuple[float, float, float]:
    return {
        "x0": (1.0, 0.0, 0.0), "xL": (-1.0, 0.0, 0.0),
        "y0": (0.0, 1.0, 0.0), "yW": (0.0, -1.0, 0.0),
        "z0": (0.0, 0.0, 1.0), "zH": (0.0, 0.0, -1.0),
    }[wall]

def terminal_direction(t: Terminal) -> tuple[float, float, float]:
    if t.dir_x is None or t.dir_y is None or t.dir_z is None:
        return normal_inward(t.wall)
    vec = (float(t.dir_x), float(t.dir_y), float(t.dir_z))
    n = math.sqrt(sum(v*v for v in vec))
    if n < 1e-9:
        return normal_inward(t.wall)
    vec = tuple(v/n for v in vec)
    inward = normal_inward(t.wall)
    dot = sum(vec[i] * inward[i] for i in range(3))
    if dot <= 0.03:
        raise HTTPException(400, f"La dirección de {t.name} debe apuntar hacia el interior del recinto.")
    return vec

def terminal_speed(t: Terminal) -> float:
    if t.speed:
        return float(t.speed)
    return t.q / 3600.0 / (t.width * t.height)

def validate_project(project: CFDRequest) -> None:
    room = project.room
    supplies = [t for t in project.terminals if t.kind == "supply"]
    returns = [t for t in project.terminals if t.kind == "return"]
    if not supplies:
        raise HTTPException(400, "Añade al menos una impulsión.")
    if not returns:
        raise HTTPException(400, "Añade al menos un retorno/extracción.")
    ids = set()
    for t in project.terminals:
        if t.id in ids:
            raise HTTPException(400, f"ID de terminal repetido: {t.id}")
        ids.add(t.id)
        _, a_axis, b_axis = wall_axes(t.wall)
        a_len = axis_length(room, a_axis)
        b_len = axis_length(room, b_axis)
        a0, a1, b0, b1 = terminal_limits(t)
        eps = 1e-7
        if a0 <= eps or a1 >= a_len - eps or b0 <= eps or b1 >= b_len - eps:
            raise HTTPException(400, f"{t.name} debe quedar completamente dentro de la superficie {t.wall}.")
        if t.kind == "supply":
            v = terminal_speed(t)
            if v <= 0 or v > 30:
                raise HTTPException(400, f"Velocidad no válida en {t.name}: {v:.2f} m/s.")
            terminal_direction(t)
    for i, ta in enumerate(project.terminals):
        for tb in project.terminals[i+1:]:
            if ta.wall != tb.wall:
                continue
            a0,a1,b0,b1 = terminal_limits(ta)
            c0,c1,d0,d1 = terminal_limits(tb)
            overlap = min(a1,c1) - max(a0,c0) > 1e-8 and min(b1,d1) - max(b0,d0) > 1e-8
            if overlap:
                raise HTTPException(400, f"Los terminales {ta.name} y {tb.name} se solapan en {ta.wall}.")

def foam_header(name: str, cls: str = "dictionary") -> str:
    return f"FoamFile\n{{ version 2.0; format ascii; class {cls}; object {name}; }}\n"

def uniq(values: List[float]) -> List[float]:
    return sorted(set(round(float(v), 9) for v in values))

def mesh_axes(project: CFDRequest):
    r = project.room
    xs, ys, zs = [0.0, r.L], [0.0, r.W], [0.0, r.H]
    for t in project.terminals:
        a0,a1,b0,b1 = terminal_limits(t)
        _, a_axis, b_axis = wall_axes(t.wall)
        axes = [xs, ys, zs]
        axes[a_axis].extend([a0,a1]); axes[b_axis].extend([b0,b1])
    return uniq(xs), uniq(ys), uniq(zs)

def patch_name(t: Terminal) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", t.id)
    return ("inlet_" if t.kind == "supply" else "outlet_") + safe

def terminal_on_face(project: CFDRequest, wall: Wall, c1: float, c2: float) -> Optional[Terminal]:
    for t in project.terminals:
        if t.wall != wall:
            continue
        a0,a1,b0,b1 = terminal_limits(t)
        if a0 < c1 < a1 and b0 < c2 < b1:
            return t
    return None

def estimate_cells(project: CFDRequest) -> int:
    xs,ys,zs = mesh_axes(project)
    total = 0
    for i in range(len(xs)-1):
        nx=max(2,math.ceil((xs[i+1]-xs[i])/DX))
        for j in range(len(ys)-1):
            ny=max(2,math.ceil((ys[j+1]-ys[j])/DY))
            for k in range(len(zs)-1):
                nz=max(2,math.ceil((zs[k+1]-zs[k])/DZ))
                total += nx*ny*nz
    return total

def build_case(project: CFDRequest, case_dir: Path) -> None:
    validate_project(project)
    for sub in ("0", "constant", "system"):
        (case_dir/sub).mkdir(parents=True, exist_ok=True)
    xs,ys,zs = mesh_axes(project)
    vertices=[]; blocks=[]
    patches: Dict[str, List[str]] = {patch_name(t): [] for t in project.terminals}; patches["walls"]=[]
    nxv, nyv = len(xs), len(ys)
    def vid(i,j,k): return i + nxv*(j + nyv*k)
    for z in zs:
        for y in ys:
            for x in xs:
                vertices.append(f"({x} {y} {z})")
    def add_face(wall, face, c1, c2):
        t=terminal_on_face(project,wall,c1,c2); patches[patch_name(t) if t else "walls"].append(face)
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            for k in range(len(zs)-1):
                x0,x1=xs[i],xs[i+1]; y0,y1=ys[j],ys[j+1]; z0,z1=zs[k],zs[k+1]
                a=vid(i,j,k); b=vid(i+1,j,k); c=vid(i+1,j+1,k); d=vid(i,j+1,k)
                e=vid(i,j,k+1); f=vid(i+1,j,k+1); g=vid(i+1,j+1,k+1); h=vid(i,j+1,k+1)
                cx=(x0+x1)/2; cy=(y0+y1)/2; cz=(z0+z1)/2
                nx=max(2,math.ceil((x1-x0)/DX)); ny=max(2,math.ceil((y1-y0)/DY)); nz=max(2,math.ceil((z1-z0)/DZ))
                blocks.append(f"hex ({a} {b} {c} {d} {e} {f} {g} {h}) ({nx} {ny} {nz}) simpleGrading (1 1 1)")
                if i==0: add_face("x0",f"({a} {e} {h} {d})",cy,cz)
                if i==len(xs)-2: add_face("xL",f"({b} {c} {g} {f})",cy,cz)
                if j==0: add_face("y0",f"({a} {b} {f} {e})",cx,cz)
                if j==len(ys)-2: add_face("yW",f"({d} {h} {g} {c})",cx,cz)
                if k==0: add_face("z0",f"({a} {d} {c} {b})",cx,cy)
                if k==len(zs)-2: add_face("zH",f"({e} {f} {g} {h})",cx,cy)
    boundary=[]
    for name,faces in patches.items():
        if faces:
            boundary.append(f"{name} {{ type {'wall' if name=='walls' else 'patch'}; faces ( {' '.join(faces)} ); }}")
    (case_dir/"system/blockMeshDict").write_text(foam_header("blockMeshDict")+"convertToMeters 1;\nvertices (\n"+"\n".join(vertices)+"\n);\nblocks (\n"+"\n".join(blocks)+"\n);\nedges ();\nboundary (\n"+"\n".join(boundary)+"\n);\nmergePatchPairs ();\n")
    supplies=[t for t in project.terminals if t.kind=="supply"]
    reference_speed=max(terminal_speed(t) for t in supplies); ref_dim=max(min(t.width,t.height) for t in supplies)
    ref_k=1.5*(reference_speed*0.05)**2; ref_eps=0.09**0.75*ref_k**1.5/max(0.07*ref_dim,1e-6)
    field_bc={"U":[],"p":[],"k":[],"epsilon":[],"nut":[]}
    for t in project.terminals:
        pn=patch_name(t)
        if t.kind=="supply":
            sp=terminal_speed(t); dx,dy,dz=terminal_direction(t)
            kval=1.5*(sp*0.05)**2; dh=2*t.width*t.height/(t.width+t.height); eps=0.09**0.75*kval**1.5/max(0.07*dh,1e-6)
            field_bc["U"].append(f"{pn} {{ type fixedValue; value uniform ({sp*dx} {sp*dy} {sp*dz}); }}")
            field_bc["p"].append(f"{pn} {{ type zeroGradient; }}")
            field_bc["k"].append(f"{pn} {{ type fixedValue; value uniform {kval}; }}")
            field_bc["epsilon"].append(f"{pn} {{ type fixedValue; value uniform {eps}; }}")
            field_bc["nut"].append(f"{pn} {{ type calculated; value uniform 0; }}")
        else:
            field_bc["U"].append(f"{pn} {{ type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0); }}")
            field_bc["p"].append(f"{pn} {{ type fixedValue; value uniform 0; }}")
            field_bc["k"].append(f"{pn} {{ type inletOutlet; inletValue uniform {ref_k}; value uniform {ref_k}; }}")
            field_bc["epsilon"].append(f"{pn} {{ type inletOutlet; inletValue uniform {ref_eps}; value uniform {ref_eps}; }}")
            field_bc["nut"].append(f"{pn} {{ type calculated; value uniform 0; }}")
    field_bc["U"].append("walls { type noSlip; }"); field_bc["p"].append("walls { type zeroGradient; }")
    field_bc["k"].append(f"walls {{ type kqRWallFunction; value uniform {ref_k}; }}")
    field_bc["epsilon"].append(f"walls {{ type epsilonWallFunction; value uniform {ref_eps}; }}")
    field_bc["nut"].append("walls { type nutkWallFunction; value uniform 0; }")
    def write_field(name,dim,internal,vector=False):
        cls="volVectorField" if vector else "volScalarField"
        (case_dir/"0"/name).write_text(foam_header(name,cls)+f"dimensions {dim};\ninternalField uniform {internal};\nboundaryField {{\n"+"\n".join(field_bc[name])+"\n}\n")
    write_field("U","[0 1 -1 0 0 0 0]","(0 0 0)",True); write_field("p","[0 2 -2 0 0 0 0]","0")
    write_field("k","[0 2 -2 0 0 0 0]",str(ref_k)); write_field("epsilon","[0 2 -3 0 0 0 0]",str(ref_eps)); write_field("nut","[0 2 -1 0 0 0 0]","0")
    (case_dir/"constant/transportProperties").write_text(foam_header("transportProperties")+"transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] 1.5e-5;\n")
    (case_dir/"constant/turbulenceProperties").write_text(foam_header("turbulenceProperties")+"simulationType RAS;\nRAS { RASModel kEpsilon; turbulence on; printCoeffs on; }\n")
    (case_dir/"system/controlDict").write_text(foam_header("controlDict")+"application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 2000; deltaT 1; writeControl timeStep; writeInterval 100; purgeWrite 2; writeFormat ascii; writePrecision 8; runTimeModifiable true;\n")
    (case_dir/"system/fvSchemes").write_text(foam_header("fvSchemes")+"ddtSchemes { default steadyState; } gradSchemes { default Gauss linear; } divSchemes { default none; div(phi,U) bounded Gauss upwind; div(phi,k) bounded Gauss upwind; div(phi,epsilon) bounded Gauss upwind; div((nuEff*dev2(T(grad(U))))) Gauss linear; } laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; } snGradSchemes { default corrected; } wallDist { method meshWave; } fluxRequired { default no; p; }\n")
    (case_dir/"system/fvSolution").write_text(foam_header("fvSolution")+'solvers { p { solver GAMG; tolerance 1e-8; relTol 0.01; smoother GaussSeidel; } "(U|k|epsilon)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; } } SIMPLE { nNonOrthogonalCorrectors 1; residualControl { p 1e-5; U 1e-5; k 1e-5; epsilon 1e-5; } } relaxationFactors { fields { p 0.3; } equations { U 0.7; k 0.7; epsilon 0.7; } }\n')

def read_tail(path: Path, chars: int=MAX_LOG_CHARS) -> str:
    return path.read_text(errors="ignore")[-chars:] if path.exists() else ""

def foam_shell(command: str, cwd: Path, job_id: str, log_name: str) -> None:
    log_path=cwd/log_name; script=f"source {FOAM_BASHRC} && {command}"
    with log_path.open("w") as log:
        proc=subprocess.Popen(["bash","-lc",script],cwd=cwd,stdout=log,stderr=subprocess.STDOUT,text=True)
        with jobs_lock: active_processes[job_id]=proc
        started=time.time()
        while proc.poll() is None:
            if time.time()-started>MAX_RUNTIME_SECONDS:
                proc.terminate(); raise RuntimeError(f"El cálculo superó el límite de {MAX_RUNTIME_SECONDS//60} min.")
            with jobs_lock:
                if jobs.get(job_id,{}).get("cancel_requested"):
                    proc.terminate(); raise RuntimeError("Cálculo cancelado por el usuario.")
            time.sleep(.5)
        with jobs_lock: active_processes.pop(job_id,None)
        if proc.returncode!=0: raise RuntimeError(f"Falló {command} (código {proc.returncode}).\n{read_tail(log_path,7000)}")

def latest_time_dir(case_dir: Path) -> Path:
    numeric=[]
    for d in case_dir.iterdir():
        if d.is_dir():
            try: numeric.append((float(d.name),d))
            except ValueError: pass
    numeric=[x for x in numeric if x[0]>0]
    if not numeric: raise RuntimeError("OpenFOAM no escribió un tiempo de solución.")
    return max(numeric,key=lambda x:x[0])[1]

def parse_vectors(path: Path) -> List[List[float]]:
    text=path.read_text(errors="ignore")
    marker=re.search(r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;",text,re.S)
    if not marker: raise RuntimeError(f"No se pudo leer el campo vectorial {path.name}.")
    rows=[[float(a),float(b),float(c)] for a,b,c in re.findall(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",marker.group(2))]
    if len(rows)!=int(marker.group(1)): raise RuntimeError(f"Campo {path.name} incompleto.")
    return rows

def parse_solver_log(log_text: str):
    its=[int(x) for x in re.findall(r"Time =\s*([0-9]+)",log_text)]
    converged="SIMPLE solution converged" in log_text or "solution converged" in log_text.lower()
    residuals={}
    for field in ("Ux","Uy","Uz","p","k","epsilon"):
        matches=re.findall(rf"Solving for {re.escape(field)}, Initial residual = ([0-9.eE+-]+)",log_text)
        if matches:
            try: residuals[field]=float(matches[-1])
            except ValueError: pass
    return (max(its) if its else 0),converged,residuals

def downsample(rows: List[List[float]], max_points: int):
    if len(rows)<=max_points: return rows
    step=math.ceil(len(rows)/max_points); return rows[::step]

def make_plane(centres,velocities,axis:int,target:float):
    if not centres: return {"position":target,"points":[]}
    positions=sorted(set(round(c[axis],9) for c in centres)); pos=min(positions,key=lambda x:abs(x-target)); points=[]
    for c,u in zip(centres,velocities):
        if abs(c[axis]-pos)<1e-7: points.append([*c,*u])
    return {"position":pos,"points":downsample(points,MAX_PLANE_POINTS)}

def make_volume(centres,velocities):
    rows=[]
    for c,u in zip(centres,velocities):
        s=math.sqrt(u[0]**2+u[1]**2+u[2]**2); rows.append([*c,*u,s])
    return {"points":downsample(rows,MAX_VOLUME_POINTS),"sourceCells":len(rows)}

def engine_version() -> str:
    try:
        out=subprocess.run(["bash","-lc",f"source {FOAM_BASHRC} >/dev/null 2>&1 && printf '%s' \"$WM_PROJECT_VERSION\""],capture_output=True,text=True,timeout=8)
        return out.stdout.strip() or "desconocida"
    except Exception: return "no disponible"

def foam_environment() -> dict:
    if not Path(FOAM_BASHRC).exists(): return {"ready":False,"version":"no disponible","missing":["OpenFOAM bashrc"]}
    required=["blockMesh","checkMesh","simpleFoam","postProcess"]
    try:
        cmd=f"source {FOAM_BASHRC} >/dev/null 2>&1 && printf '%s\\n' \"$WM_PROJECT_VERSION\" && "+" && ".join(f"command -v {x} >/dev/null" for x in required)
        out=subprocess.run(["bash","-lc",cmd],text=True,capture_output=True,timeout=10)
        if out.returncode!=0: return {"ready":False,"version":engine_version(),"missing":required}
        return {"ready":True,"version":((out.stdout.strip().splitlines() or [engine_version()])[0]),"missing":[]}
    except Exception: return {"ready":False,"version":engine_version(),"missing":required}

def update_job(job_id: str, **values):
    with jobs_lock:
        if job_id in jobs: jobs[job_id].update(values)

def run_job(job_id: str, project: CFDRequest):
    acquired=False; started=time.time(); case_dir=JOBS_DIR/job_id/"irb-air"
    try:
        update_job(job_id,stage="En cola para el solver"); solver_gate.acquire(); acquired=True
        with jobs_lock:
            if jobs[job_id].get("cancel_requested"): raise RuntimeError("Cálculo cancelado por el usuario.")
        case_dir.mkdir(parents=True,exist_ok=True)
        update_job(job_id,stage="Generando caso OpenFOAM"); build_case(project,case_dir)
        update_job(job_id,stage="Generando malla"); foam_shell("blockMesh",case_dir,job_id,"log.blockMesh")
        update_job(job_id,stage="Comprobando malla"); foam_shell("checkMesh",case_dir,job_id,"log.checkMesh")
        check_text=read_tail(case_dir/"log.checkMesh",20000)
        if "Mesh OK" not in check_text: raise RuntimeError("checkMesh no ha acreditado 'Mesh OK'.")
        update_job(job_id,stage="Resolviendo RANS k-epsilon"); foam_shell("simpleFoam -noFunctionObjects",case_dir,job_id,"log.simpleFoam")
        update_job(job_id,stage="Extrayendo campo 3D"); foam_shell("postProcess -func writeCellCentres -latestTime",case_dir,job_id,"log.postProcess")
        td=latest_time_dir(case_dir); centres=parse_vectors(td/"C"); velocities=parse_vectors(td/"U")
        if len(centres)!=len(velocities): raise RuntimeError("El número de centros y velocidades no coincide.")
        slog=(case_dir/"log.simpleFoam").read_text(errors="ignore"); iteration,converged,residuals=parse_solver_log(slog)
        cm=re.search(r"cells:\s*([0-9]+)",check_text); cells=int(cm.group(1)) if cm else len(centres)
        max_speed=max((math.sqrt(sum(vv*vv for vv in u)) for u in velocities),default=0.0); r=project.room
        result={"status":"CALCULADO_SIN_VALIDACION_EXPERIMENTAL","build":BUILD,"project":project.model_dump(),"cells":cells,"iteration":iteration,"converged":converged,"residuals":residuals,"imbalancePercent":None,"planes":{"section":make_plane(centres,velocities,1,r.W/2),"plan":make_plane(centres,velocities,2,min(1.1,r.H/2))},"volume":make_volume(centres,velocities),"maxSpeed":max_speed,"engineVersion":engine_version(),"executionDate":time.strftime("%Y-%m-%d"),"solverSeconds":round(time.time()-started,2),"validation":"Campo 3D procedente de las celdas reales de OpenFOAM. La animación usa trazadores sobre el campo medio estacionario RANS; no es un transitorio CFD. CFD isotermo con simpleFoam, RANS k-epsilon. Retornos a presión relativa 0 Pa. Pendientes: caudal integrado por patch, yPlus, independencia de malla, solver térmico y validación experimental."}
        update_job(job_id,status="finished",stage="Terminado",result=result,iteration=iteration,log=read_tail(case_dir/"log.simpleFoam"))
    except Exception as exc:
        status="cancelled" if "cancelado" in str(exc).lower() else "failed"
        update_job(job_id,status=status,stage="Cancelado" if status=="cancelled" else "Error",error=str(exc),log=read_tail(case_dir/"log.simpleFoam"))
    finally:
        if acquired: solver_gate.release()

@app.get("/")
def root():
    p=PUBLIC_DIR/"index.html"
    if not p.exists(): raise HTTPException(404,"Frontend no encontrado")
    return FileResponse(p)

@app.get("/api/health")
def health():
    e=foam_environment(); return {"ok":True,"build":BUILD,"engineReady":e["ready"],"engineVersion":e["version"]}

@app.get("/api/engine")
def engine():
    e=foam_environment(); return {"ready":e["ready"],"version":e["version"],"missing":e["missing"],"build":BUILD,"api":API_VERSION}

@app.post("/api/jobs")
def create_job(req: JobRequest):
    validate_project(req.project); cells=estimate_cells(req.project)
    if cells>MAX_ESTIMATED_CELLS: raise HTTPException(400,f"Malla estimada demasiado grande: {cells:,} celdas. Límite: {MAX_ESTIMATED_CELLS:,}.")
    job_id=uuid.uuid4().hex[:16]
    with jobs_lock: jobs[job_id]={"id":job_id,"status":"queued","stage":"Preparando","estimatedCells":cells,"iteration":0,"cancel_requested":False}
    threading.Thread(target=run_job,args=(job_id,req.project),daemon=True).start()
    return {"id":job_id,"status":"queued","estimatedCells":cells}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        j=jobs.get(job_id)
        if not j: raise HTTPException(404,"Cálculo no encontrado.")
        out=dict(j)
    case_dir=JOBS_DIR/job_id/"irb-air"
    if out.get("status") not in ("finished","failed","cancelled") and (case_dir/"log.simpleFoam").exists():
        tail=read_tail(case_dir/"log.simpleFoam"); out["log"]=tail
        its=[int(x) for x in re.findall(r"Time =\s*([0-9]+)",tail)]
        if its: out["iteration"]=max(its)
    return JSONResponse(out)

@app.post("/api/cancel")
def cancel(req: CancelRequest):
    with jobs_lock:
        if req.id not in jobs: raise HTTPException(404,"Cálculo no encontrado.")
        jobs[req.id]["cancel_requested"]=True; proc=active_processes.get(req.id)
    if proc and proc.poll() is None: proc.terminate()
    return {"ok":True}
