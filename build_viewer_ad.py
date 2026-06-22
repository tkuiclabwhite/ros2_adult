#!/usr/bin/env python3
"""
AD Adult Robot 3D Viewer - Build Script
用法：python3 build_viewer_ad.py
"""

import base64, json, pathlib, xml.etree.ElementTree as ET

HOME      = pathlib.Path.home()
URDF_PATH = HOME / "AD/urdf/AD.urdf"
MESH_DIR  = HOME / "AD/meshes"
OUTPUT    = HOME / "ros2_adult/hurocup_interface/RobotViewer.html"
STAND_INI = HOME / "ros2_adult/src/strategy/strategy/Parameter/stand.ini"

THREEJS   = HOME / "ros2_adult/hurocup_interface/js/three.min.js"
EE2_JS    = HOME / "ros2_adult/hurocup_interface/js/roslib/eventemitter2.min.js"
ROSLIB_JS = HOME / "ros2_adult/hurocup_interface/js/roslib/roslib.min.js"


def _floats(s, default="0 0 0"):
    return [float(v) for v in (s or default).split()]

def parse_urdf(path):
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        name = link.get("name")
        mesh_file, vo = None, {"xyz": [0,0,0], "rpy": [0,0,0]}
        visual = link.find("visual")
        if visual is not None:
            o = visual.find("origin")
            if o is not None:
                vo = {"xyz": _floats(o.get("xyz")), "rpy": _floats(o.get("rpy"))}
            m = visual.find("geometry/mesh")
            if m is not None:
                mesh_file = pathlib.Path(m.get("filename", "")).name
        links[name] = {"mesh": mesh_file, "vo": vo}
    joints = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        p  = joint.find("parent"); c  = joint.find("child")
        o  = joint.find("origin"); ax = joint.find("axis")
        origin = {"xyz": [0,0,0], "rpy": [0,0,0]}
        if o is not None:
            origin = {"xyz": _floats(o.get("xyz")), "rpy": _floats(o.get("rpy"))}
        axis = _floats(ax.get("xyz") if ax is not None else None, "1 0 0")
        joints[name] = {
            "type":   joint.get("type"),
            "parent": p.get("link") if p is not None else None,
            "child":  c.get("link") if c is not None else None,
            "origin": origin,
            "axis":   axis,
        }
    return links, joints

def load_meshes(mesh_dir, links):
    meshes, seen = {}, set()
    for data in links.values():
        fname = data["mesh"]
        if not fname or fname in seen: continue
        fpath = mesh_dir / fname
        if fpath.exists():
            raw = fpath.read_bytes()
            meshes[fname] = base64.b64encode(raw).decode()
            seen.add(fname)
            print(f"  ✓ {fname:35s} {len(raw)//1024:4d} KB")
        else:
            print(f"  ✗ {fname} 找不到")
    return meshes

def read_js(path):
    if path.exists(): return path.read_text(encoding="utf-8")
    return f"/* 找不到 {path.name} */"

def parse_stand_ini(path):
    """從 stand.ini 讀取 motionstate=3 的馬達刻度 (1-based dict)"""
    stand_real = {}
    try:
        text = path.read_text(encoding="utf-8")
        in_target = False
        for line in text.splitlines():
            line = line.strip()
            if line == "motionstate = 3":
                in_target = True
            elif line.startswith("motionstate") and in_target:
                break
            elif in_target and line.startswith("motordata"):
                ticks = [int(x) for x in line.split("=", 1)[1].split(",")]
                for i, t in enumerate(ticks):
                    stand_real[i + 1] = t  # 1-based motor ID
                break
        if stand_real:
            print(f"  ✓ stand.ini loaded: {len(stand_real)} motors from {path}")
        else:
            print(f"  ✗ stand.ini: motionstate=3 not found in {path}")
    except Exception as e:
        print(f"  ✗ stand.ini read error: {e}")
    return stand_real

def generate_html(links, joints, meshes, three_js, ee2_js, roslib_js):
    links_json  = json.dumps(links,  separators=(",", ":"))
    joints_json = json.dumps(joints, separators=(",", ":"))
    meshes_json = json.dumps(meshes, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="utf-8">
<title>AD Robot Viewer</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#12121f;color:#ddd;font-family:monospace;overflow:hidden}}
canvas{{display:block}}
#hud{{position:absolute;top:12px;left:12px;background:rgba(0,0,0,.65);
      padding:8px 14px;border-radius:6px;font-size:12px;line-height:1.8;
      pointer-events:none}}
.dot-on{{color:#4caf50}}.dot-off{{color:#f44336}}.dot-warn{{color:#ff9800}}
#ctrl-bar{{position:absolute;top:12px;right:12px;display:flex;gap:6px;align-items:center}}
#ctrl-bar select,#ctrl-bar button{{
  height:28px;padding:0 10px;border:none;border-radius:4px;
  font-size:12px;cursor:pointer}}
#ctrl-bar select{{background:#1e2a3a;color:#ddd}}
#btn-stand{{background:#1a4a2a;color:#adf}}
#btn-stand:hover{{background:#1f6035}}
</style>
</head>
<body>
<div id="hud">
  <div>AD Robot Viewer</div>
  <div id="ros-line"><span id="ros-dot" class="dot-off">●</span> <span id="ros-txt">未連線 rosbridge</span></div>
  <div id="js-line"><span id="js-dot" class="dot-off">●</span> <span id="js-txt">/joint_commands: 等待中</span></div>
  <div id="stand-line"><span id="stand-dot" class="dot-warn">●</span> <span id="stand-txt">stand.ini: 使用預設值</span></div>
</div>
<div id="ctrl-bar">
  <select id="addressSelect">
    <option value="172.17.121.10">172.17.121.10</option>
    <option value="localhost">localhost</option>
  </select>
  <button type="button" style="width:110px;height:25px;" onclick="enterAddress()">Enter Address</button>
  <button id="btn-stand" style="height:25px;" onclick="reloadStandIni()">重讀 stand.ini</button>
  <button type="button" style="height:25px;background:#2a3a5a;color:#adf;border:none;border-radius:4px;padding:0 10px;font-size:12px;cursor:pointer;" onclick="openSGPPanel()">調整 STAND_GP</button>
</div>

<!-- STAND_GP 側邊欄 -->
<div id="sgp-panel" style="position:fixed;top:0;right:-260px;width:250px;height:100%;
  background:#1a2030;color:#ddd;border-left:1px solid #446;z-index:9999;
  overflow-y:auto;padding:12px 10px;font-size:12px;
  transition:right 0.2s ease;box-shadow:-4px 0 16px #0009;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <b style="font-size:13px;">STAND_GP 理想刻度</b>
    <button onclick="closeSGPPanel()" style="background:none;border:none;color:#aaa;font-size:16px;cursor:pointer;">✕</button>
  </div>
  <div id="sgp-grid" style="display:flex;flex-direction:column;gap:5px;"></div>
  <div style="margin-top:12px;">
    <button onclick="applySGP()" style="width:100%;height:28px;background:#2255aa;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:12px;">套用</button>
  </div>
</div>

<script>{three_js}</script>
<script>{ee2_js}</script>
<script>{roslib_js}</script>
<script>
// ════════════════════════════════════════════════════════════
//  嵌入資料
// ════════════════════════════════════════════════════════════
const URDF_LINKS  = {links_json};
const URDF_JOINTS = {joints_json};
const MESH_DATA   = {meshes_json};

// ════════════════════════════════════════════════════════════
//  換算常數
// ════════════════════════════════════════════════════════════
const TPR    = 4096.0 / (2 * Math.PI);
const CENTER = 2048;

// 左手1-7，右手8-14，頭28-29
const JOINT_TO_MOTOR = {{
  LshouderJoint:1, LarmJoint:2,  Larm1Joint:3,
  LforarmJoint:4,  Lforarm1Joint:5, Lforarm2Joint:6, LhandJoint:7,
  RshouderJoint:8, RarmJoint:9,  Rarm1Joint:10,
  RforarmJoint:11, Rforarm1Joint:12, Rforarm2Joint:13, RhandJoint:14,
  NeckJoint:28, HeadJoint:29,
}};

// 方向（先全設 1，依實際馬達調整）
const DIR = {{
  1:-1,2:1,3:-1,4:1,5:1,6:-1,7:-1,
  8:-1,9:1,10:-1,11:1,12:1,13:-1,14:-1,
  28:1,29:1,
}};

const JOINT_OFFSET = {{}};

const STAND_GP = {{
  1:2048,2:1792,3:3072,4:2304,5:1024,6:2048,7:2048,
  8:2048,9:2304,10:3072,11:1792,12:1024,13:2048,14:2048,
  28:2048,29:2048,
}};

let STAND_REAL = Object.assign({{}}, STAND_GP);

function ticksToRad(tick, motorId) {{
  const dir       = DIR[motorId]          ?? 1;
  const offset    = JOINT_OFFSET[motorId] ?? 0;
  const standReal = STAND_REAL[motorId]   ?? CENTER;
  const standGp   = STAND_GP[motorId]     ?? CENTER;
  const corrected = tick - standReal + standGp;
  return dir * (corrected - CENTER) / TPR + offset;
}}

// ════════════════════════════════════════════════════════════
//  Three.js 初始化
// ════════════════════════════════════════════════════════════
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x12121f);
scene.add(new THREE.GridHelper(3, 24, 0x334466, 0x223355));

const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.005, 50);

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(2, 5, 3); sun.castShadow = true; scene.add(sun);
const fill = new THREE.DirectionalLight(0x8899ff, 0.3);
fill.position.set(-2, 3, -2); scene.add(fill);

// 簡易 Orbit 控制（左鍵旋轉，右/中鍵平移）
let drag=false, pan=false, lastX=0, lastY=0;
const sph = {{ theta: Math.PI, phi: 1.0, r: 2.2 }};
const tgt = new THREE.Vector3(0, 0.8, 0);
function updateCam() {{
  camera.position.set(
    tgt.x + sph.r*Math.sin(sph.phi)*Math.sin(sph.theta),
    tgt.y + sph.r*Math.cos(sph.phi),
    tgt.z + sph.r*Math.sin(sph.phi)*Math.cos(sph.theta)
  );
  camera.lookAt(tgt);
}}
updateCam();
const cvs = renderer.domElement;
cvs.addEventListener('contextmenu', e => e.preventDefault());
cvs.addEventListener('mousedown', e => {{
  if (e.button===2||e.button===1) {{ pan=true; e.preventDefault(); }} else drag=true;
  lastX=e.clientX; lastY=e.clientY;
}});
cvs.addEventListener('mouseup',    () => {{ drag=false; pan=false; }});
cvs.addEventListener('mouseleave', () => {{ drag=false; pan=false; }});
cvs.addEventListener('mousemove', e => {{
  if (!drag&&!pan) return;
  const dx=e.clientX-lastX, dy=e.clientY-lastY;
  if (pan) {{
    const ps=sph.r*0.001;
    const rx=Math.cos(sph.theta), rz=-Math.sin(sph.theta);
    const ux=-Math.sin(sph.theta)*Math.cos(sph.phi), uy=Math.sin(sph.phi), uz=-Math.cos(sph.theta)*Math.cos(sph.phi);
    tgt.x -= dx*ps*rx-dy*ps*ux; tgt.y += dy*ps*uy; tgt.z -= dx*ps*rz-dy*ps*uz;
  }} else {{
    sph.theta -= dx*0.008;
    sph.phi = Math.max(0.05, Math.min(Math.PI-0.05, sph.phi-dy*0.008));
  }}
  lastX=e.clientX; lastY=e.clientY; updateCam();
}});
cvs.addEventListener('wheel', e => {{
  sph.r = Math.max(0.2, Math.min(8, sph.r*(1+e.deltaY*0.001)));
  updateCam(); e.preventDefault();
}}, {{passive:false}});
window.addEventListener('resize', () => {{
  camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});

// ════════════════════════════════════════════════════════════
//  STL 解析
// ════════════════════════════════════════════════════════════
function base64ToBuffer(b64) {{
  const bin=atob(b64), buf=new ArrayBuffer(bin.length), v=new Uint8Array(buf);
  for (let i=0;i<bin.length;i++) v[i]=bin.charCodeAt(i); return buf;
}}
function parseBinarySTL(buf) {{
  const view=new DataView(buf), n=view.getUint32(80,true);
  const pos=new Float32Array(n*9); let off=84;
  for (let i=0;i<n;i++) {{
    off+=12;
    for (let j=0;j<9;j++) {{ pos[i*9+j]=view.getFloat32(off,true); off+=4; }}
    off+=2;
  }}
  return pos;
}}
function makeMesh(filename) {{
  const b64=MESH_DATA[filename]; if (!b64) return null;
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(parseBinarySTL(base64ToBuffer(b64)),3));
  geo.computeVertexNormals();
  const mat=new THREE.MeshPhongMaterial({{color:0x3a7bd5,specular:0x1a2a4a,shininess:40,side:THREE.DoubleSide}});
  const m=new THREE.Mesh(geo,mat); m.castShadow=true; return m;
}}

// ════════════════════════════════════════════════════════════
//  URDF Scene graph
// ════════════════════════════════════════════════════════════
const linkObjects={{}}, jointPivots={{}};
function setRPY(obj, rpy) {{ obj.rotation.set(rpy[0],rpy[1],rpy[2],'ZYX'); }}

function buildLink(linkName, parentGroup) {{
  const grp=new THREE.Group(); grp.name=linkName;
  linkObjects[linkName]=grp; parentGroup.add(grp);
  const ld=URDF_LINKS[linkName];
  if (ld?.mesh) {{
    const mesh=makeMesh(ld.mesh);
    if (mesh) {{ mesh.position.set(...ld.vo.xyz); setRPY(mesh,ld.vo.rpy); grp.add(mesh); }}
  }}
  for (const [jname,jd] of Object.entries(URDF_JOINTS)) {{
    if (jd.parent!==linkName) continue;
    const offsetGrp=new THREE.Group(); offsetGrp.name=jname+'_offset';
    offsetGrp.position.set(...jd.origin.xyz); setRPY(offsetGrp,jd.origin.rpy);
    grp.add(offsetGrp);
    const pivot=new THREE.Group(); pivot.name=jname+'_pivot';
    offsetGrp.add(pivot); jointPivots[jname]=pivot;
    if (jd.child) buildLink(jd.child,pivot);
  }}
}}

const worldGroup=new THREE.Group();
worldGroup.rotation.x=Math.PI/2;
worldGroup.position.y=1.1;
scene.add(worldGroup);

const childSet=new Set(Object.values(URDF_JOINTS).map(j=>j.child).filter(Boolean));
const rootLink=Object.keys(URDF_LINKS).find(n=>!childSet.has(n));
buildLink(rootLink, worldGroup);
console.log('根節點:', rootLink, '關節數:', Object.keys(jointPivots).length);

function centerOnGrid() {{
  worldGroup.updateWorldMatrix(true,true);
  const box=new THREE.Box3().setFromObject(worldGroup);
  const c=new THREE.Vector3(); box.getCenter(c);
  worldGroup.position.x-=c.x; worldGroup.position.z-=c.z;
}}

// ════════════════════════════════════════════════════════════
//  關節角度 / 動畫狀態
// ════════════════════════════════════════════════════════════
function setJointAngle(jname, rad) {{
  const pivot=jointPivots[jname]; if (!pivot) return;
  pivot.setRotationFromAxisAngle(new THREE.Vector3(...URDF_JOINTS[jname].axis).normalize(), rad);
}}

var animCurrent={{}}, animTarget={{}};

function applyIdealStandPose() {{
  for (const [jname,mid] of Object.entries(JOINT_TO_MOTOR)) {{
    const tick=STAND_GP[mid]??CENTER;
    const rad=(DIR[mid]??1)*(tick-CENTER)/TPR+(JOINT_OFFSET[mid]??0);
    animCurrent[jname]=rad; animTarget[jname]={{rad,vel:0}};
    setJointAngle(jname,rad);
  }}
}}

function applyTickMap(tickMap, velMap={{}}) {{
  for (const [jname,mid] of Object.entries(JOINT_TO_MOTOR)) {{
    if (tickMap[mid]!=null) {{
      const rad=ticksToRad(tickMap[mid],mid);
      const vel=(velMap[mid]??0)>0 ? velMap[mid]*0.02399 : 0;
      animTarget[jname]={{rad,vel}};
    }}
  }}
}}

applyIdealStandPose();
centerOnGrid();

// ════════════════════════════════════════════════════════════
//  STAND_GP 面板
// ════════════════════════════════════════════════════════════
const SGP_LABELS={{
  1:'LshouderJoint',2:'LarmJoint',3:'Larm1Joint',
  4:'LforarmJoint',5:'Lforarm1Joint',6:'Lforarm2Joint',7:'LhandJoint',
  8:'RshouderJoint',9:'RarmJoint',10:'Rarm1Joint',
  11:'RforarmJoint',12:'Rforarm1Joint',13:'Rforarm2Joint',14:'RhandJoint',
  28:'NeckJoint',29:'HeadJoint',
}};
function openSGPPanel() {{
  const panel=document.getElementById('sgp-panel');
  if (panel.style.right==='0px') {{ panel.style.right='-260px'; return; }}
  const grid=document.getElementById('sgp-grid'); grid.innerHTML='';
  const ids=Object.keys(STAND_GP).map(Number).sort((a,b)=>a-b);
  for (const mid of ids) {{
    const row=document.createElement('div');
    row.style.cssText='display:flex;align-items:center;gap:4px;';
    const lbl=SGP_LABELS[mid]||('M'+mid);
    row.innerHTML=`<span style="width:26px;color:#7af;font-weight:bold;font-size:11px;flex-shrink:0;">${{mid}}</span>`+
      `<span style="flex:1;color:#aaa;font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${{lbl}}">${{lbl}}</span>`;
    const inp=document.createElement('input');
    inp.type='number'; inp.min=0; inp.max=4096; inp.step=1; inp.value=STAND_GP[mid];
    inp.dataset.mid=mid;
    inp.style.cssText='width:58px;height:20px;background:#111;color:#fff;border:1px solid #446;border-radius:3px;text-align:center;font-size:11px;flex-shrink:0;';
    row.appendChild(inp); grid.appendChild(row);
  }}
  panel.style.right='0px';
}}
function closeSGPPanel() {{ document.getElementById('sgp-panel').style.right='-260px'; }}
function applySGP() {{
  document.querySelectorAll('#sgp-grid input[data-mid]').forEach(inp=>{{
    const mid=Number(inp.dataset.mid);
    STAND_GP[mid]=Math.max(0,Math.min(4096,parseInt(inp.value)||0));
    inp.value=STAND_GP[mid];
  }});
  applyIdealStandPose();
}}

// ════════════════════════════════════════════════════════════
//  絲滑動畫
// ════════════════════════════════════════════════════════════
const animClock=new THREE.Clock();
(function animate() {{
  requestAnimationFrame(animate);
  const dt=Math.min(animClock.getDelta(),0.05);
  for (const [jname,tgt] of Object.entries(animTarget)) {{
    const cur=animCurrent[jname]??tgt.rad;
    const diff=tgt.rad-cur;
    if (Math.abs(diff)<1e-5) continue;
    const next=tgt.vel>0
      ? (Math.abs(diff)<=tgt.vel*dt ? tgt.rad : cur+Math.sign(diff)*tgt.vel*dt)
      : tgt.rad;
    if (next!==cur) {{ animCurrent[jname]=next; setJointAngle(jname,next); }}
  }}
  renderer.render(scene,camera);
}})();

// ════════════════════════════════════════════════════════════
//  ROS 連線
// ════════════════════════════════════════════════════════════
function setHud(id, cls, txt) {{
  document.getElementById(id+'-dot').className=cls;
  document.getElementById(id+'-txt').textContent=txt;
}}

var myAddress='172.17.121.10', connectFlag=false;
const ros=new ROSLIB.Ros({{url:'ws://'+myAddress+':9090'}});
let jsSub=null;

ros.on('connection', ()=>{{
  connectFlag=true;
  setHud('ros','dot-on','已連線 '+myAddress);
  loadStandIni();
  startJointSub();
}});
ros.on('error', ()=>{{
  connectFlag=false;
  setHud('ros','dot-off','連線錯誤 ('+myAddress+')');
}});
ros.on('close', ()=>{{
  connectFlag=false;
  setHud('ros','dot-off','已斷線');
  setHud('js','dot-off','/joint_commands: 已斷線');
}});

function enterAddress() {{
  if (connectFlag) {{ try{{ros.close();}}catch(e){{}} connectFlag=false; }}
  myAddress=document.getElementById('addressSelect').value;
  setHud('ros','dot-warn','連線中 '+myAddress+'...');
  const state=ros.socket?ros.socket.readyState:WebSocket.CLOSED;
  if (state===WebSocket.OPEN||state===WebSocket.CONNECTING||state===WebSocket.CLOSING) {{
    ros.close();
    ros.once('close',()=>ros.connect('ws://'+myAddress+':9090'));
  }} else {{
    ros.connect('ws://'+myAddress+':9090');
  }}
}}

// ── stand.ini ────────────────────────────────────────────────
function loadStandIni() {{
  const svc=new ROSLIB.Service({{
    ros, name:'/package/InterfaceReadSaveMotion',
    serviceType:'tku_msgs/ReadMotion'
  }});
  svc.callService(
    new ROSLIB.ServiceRequest({{read:true,name:'stand',readstate:1}}),
    function(data) {{
      const BLOCK_SIZE=27;  // stand.ini 每個 motionstate block 的馬達數
      let absIdx=0, found=false;
      for (let i=0;i<data.vectorcnt;i++) {{
        const ms=data.motionstate[i];
        if (ms===3) {{
          // 只更新手臂馬達 1-14，15-27 不用修正，28-29 理想=真實
          for (let j=1;j<=14;j++) {{
            STAND_REAL[j]=data.absolutedata[absIdx+(j-1)];
          }}
          found=true; break;
        }} else if (ms===4) {{
          absIdx+=BLOCK_SIZE;
        }}
      }}
      if (found) {{
        setHud('stand','dot-on','stand.ini: 已載入 (motionstate=3)');
        applyIdealStandPose();
      }} else {{
        setHud('stand','dot-warn','stand.ini: 找不到 motionstate=3，使用預設');
      }}
    }},
    ()=>setHud('stand','dot-warn','stand.ini: 服務呼叫失敗，使用預設值')
  );
}}

function reloadStandIni() {{
  if (connectFlag) {{
    setHud('stand','dot-warn','stand.ini: 重讀中...');
    loadStandIni();
  }} else {{
    setHud('stand','dot-off','stand.ini: 未連線');
  }}
}}

// ── /joint_commands ──────────────────────────────────────────
function startJointSub() {{
  if (jsSub) jsSub.unsubscribe();
  jsSub=new ROSLIB.Topic({{
    ros, name:'/joint_commands', messageType:'sensor_msgs/JointState'
  }});
  let lastTime=0;
  jsSub.subscribe(msg=>{{
    const now=Date.now();
    if (now-lastTime<50) return;  // 限流 20 Hz
    lastTime=now;
    const ticks={{}},vels={{}};
    for (let i=0;i<msg.name.length;i++) {{
      const id=parseInt(msg.name[i]);
      if (!isNaN(id)) {{
        ticks[id]=Math.round(msg.position[i]);
        if (msg.velocity&&msg.velocity[i]) vels[id]=msg.velocity[i];
      }}
    }}
    applyTickMap(ticks,vels);
    setHud('js','dot-on','/joint_commands: 接收中');
  }});
  setHud('js','dot-warn','/joint_commands: 接收中');
}}

setHud('ros','dot-warn','連線中 '+myAddress+'...');
</script>
</body>
</html>"""


def main():
    print("AD Robot Viewer Builder")
    print(f"URDF: {URDF_PATH}")
    if not URDF_PATH.exists():
        print("錯誤：找不到 URDF"); return
    links, joints = parse_urdf(URDF_PATH)
    print(f"解析完成：{len(links)} links, {len(joints)} joints")
    print("載入 STL meshes:")
    meshes = load_meshes(MESH_DIR, links)
    html = generate_html(links, joints, meshes, read_js(THREEJS), read_js(EE2_JS), read_js(ROSLIB_JS))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"\n✓ 輸出：{OUTPUT}")
    print(f"  檔案大小：{OUTPUT.stat().st_size/1024/1024:.1f} MB")

if __name__ == "__main__":
    main()
