// Nova 2 digital twin -- browser side.
// Draws Dobot's Nova 2 model (URDF chain + STL meshes) at the joint angles the
// controller reports (pushed by digital_twin.py 30 times a second), and sends
// commands back; the server checks them before anything reaches the robot.

import * as THREE from 'three';
import { OrbitControls } from './vendor/three/OrbitControls.js';
import { STLLoader } from './vendor/three/STLLoader.js';
import { TransformControls } from './vendor/three/TransformControls.js';

const DEG = Math.PI / 180;
const $ = (id) => document.getElementById(id);
const fromRows = (rows) => new THREE.Matrix4().set(...rows.flat());
const toRows = (m) => {
  const e = m.elements;
  return [[e[0], e[4], e[8], e[12]], [e[1], e[5], e[9], e[13]], [e[2], e[6], e[10], e[14]], [0, 0, 0, 1]];
};
const mm = (v) => (v * 1000).toFixed(1);

let ws = null, model = null, config = null, state = null, reqId = 0;
const pending = new Map();
let robot = null, ghost = null, sceneReady = false;
let pickMode = null;              // 'object' | 'place' | null
let followCamera = false;
const edited = new Array(6).fill(false);
const targets = new Array(6).fill(0);

// ============================================================
// SCENE
// ============================================================

const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#0f141b');
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20);
camera.up.set(0, 0, 1);
camera.position.set(-1.1, -1.0, 0.85);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(-0.2, 0.05, 0.15);
orbit.enableDamping = true;
orbit.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0x3a4250, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(-0.8, -1.2, 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1, near: 0.1, far: 6 });
scene.add(sun);
scene.add(new THREE.AxesHelper(0.12));

const world = { table: null, objectMesh: null, placeMarker: null, path: null, waypoints: new THREE.Group(), trail: null };
scene.add(world.waypoints);

function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(view);

const stlLoader = new STLLoader();
const geometries = new Map();
function loadGeometry(url) {
  if (!geometries.has(url)) {
    geometries.set(url, new Promise((resolve, reject) => stlLoader.load(url, resolve, undefined, reject)));
  }
  return geometries.get(url);
}

function buildTable(tableZ) {
  const group = new THREE.Group();
  const top = new THREE.Mesh(
    new THREE.PlaneGeometry(1.8, 1.6),
    new THREE.MeshStandardMaterial({ color: 0x46505e, roughness: 0.9 }));
  top.position.set(-0.2, 0.1, tableZ);
  top.receiveShadow = true;
  top.name = 'table';
  group.add(top);
  const grid = new THREE.GridHelper(1.6, 16, 0x6b7a90, 0x56637a);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(-0.2, 0.1, tableZ + 0.0005);
  group.add(grid);
  // the Nova 2's 625 mm reach around its base
  const reach = new THREE.Mesh(new THREE.RingGeometry(0.62, 0.625, 160),
    new THREE.MeshBasicMaterial({ color: 0x4f8cff, transparent: true, opacity: 0.55 }));
  reach.position.z = tableZ + 0.001;
  group.add(reach);
  if (tableZ < -0.002) {   // the robot stands on a pedestal above the table
    const h = -tableZ;
    const ped = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.085, h, 48),
      new THREE.MeshStandardMaterial({ color: 0x2b313b, roughness: 0.7 }));
    ped.rotation.x = Math.PI / 2;
    ped.position.z = tableZ + h / 2;
    ped.castShadow = true;
    group.add(ped);
  }
  scene.add(group);
  world.table = top;
}

function buildRobot(isGhost) {
  const material = isGhost
    ? new THREE.MeshBasicMaterial({ color: 0x4fc3f7, transparent: true, opacity: 0.22, depthWrite: false })
    : null;
  const root = new THREE.Group();
  const joints = [];
  const addLink = (parent, name) => {
    const info = model.links[name];
    if (!info || !info.mesh) return;
    const [r, g, b] = info.color;
    const mat = material || new THREE.MeshStandardMaterial({ color: new THREE.Color(r, g, b), roughness: 0.45, metalness: 0.1 });
    loadGeometry(info.mesh).then((geometry) => {
      const mesh = new THREE.Mesh(geometry, mat);
      mesh.castShadow = !isGhost;
      mesh.receiveShadow = !isGhost;
      mesh.name = name;
      parent.add(mesh);
      robotMeshesLoaded += isGhost ? 0 : 1;
    });
  };
  addLink(root, model.base);
  let parent = root;
  for (const j of model.joints) {
    const origin = new THREE.Group();
    origin.matrixAutoUpdate = false;
    origin.matrix.copy(fromRows(j.origin));
    const turn = new THREE.Group();
    origin.add(turn);
    parent.add(origin);
    addLink(turn, j.child);
    joints.push(turn);
    parent = turn;
  }
  const link6 = parent;

  // gripper, drawn in the flange frame: tool tip 0.197 m out on the flange axis
  const flange = new THREE.Group();
  flange.matrixAutoUpdate = false;
  flange.matrix.copy(fromRows(model.link6_flange));
  link6.add(flange);
  const dark = material || new THREE.MeshStandardMaterial({ color: 0x30363f, roughness: 0.6, metalness: 0.3 });
  const steel = material || new THREE.MeshStandardMaterial({ color: 0x9aa3ad, roughness: 0.35, metalness: 0.6 });
  const mount = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, 0.012, 32), steel);
  mount.rotation.x = Math.PI / 2;
  mount.position.z = 0.006;
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.095, 0.05, 0.085), dark);
  body.position.z = 0.0545;
  const fingers = [-1, 1].map((side) => {
    const f = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.03, 0.1), steel);
    f.position.set(side * 0.046, 0, 0.147);
    f.userData.side = side;
    return f;
  });
  for (const m of [mount, body, ...fingers]) { m.castShadow = !isGhost; flange.add(m); }

  // wrist camera (RealSense-sized box), its optical frame: z = view axis
  const cam = new THREE.Group();
  cam.matrixAutoUpdate = false;
  cam.matrix.copy(fromRows(model.link6_camera));
  link6.add(cam);
  const camBox = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.025, 0.025), dark);
  camBox.position.z = -0.012;
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.004, 20),
    material || new THREE.MeshStandardMaterial({ color: 0x111111 }));
  lens.rotation.x = Math.PI / 2;
  cam.add(camBox, lens);

  scene.add(root);
  return {
    root, joints, fingers, camera: cam,
    setJoints(q) { q.forEach((a, i) => { joints[i].rotation.z = a * DEG; }); },
    setGrip(position) {
      const gap = 0.08 * (1 - Math.min(255, Math.max(0, position)) / 255);
      for (const f of fingers) f.position.x = f.userData.side * (gap / 2 + 0.006);
    },
  };
}
let robotMeshesLoaded = 0;

function buildObject(diameter) {
  if (world.objectMesh) scene.remove(world.objectMesh);
  const g = new THREE.Group();
  g.matrixAutoUpdate = false;
  const r = diameter / 2;
  const fruit = new THREE.Mesh(new THREE.SphereGeometry(r, 40, 24),
    new THREE.MeshStandardMaterial({ color: 0xd8432f, roughness: 0.35 }));
  fruit.scale.z = 0.88;
  fruit.castShadow = true;
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.08, r * 0.1, r * 0.35, 10),
    new THREE.MeshStandardMaterial({ color: 0x3f8a3a }));
  stem.rotation.x = Math.PI / 2;
  stem.position.z = r * 0.95;
  g.add(fruit, stem);
  g.userData.diameter = diameter;
  scene.add(g);
  world.objectMesh = g;
}

function buildPlaceMarker() {
  const g = new THREE.Group();
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.022, 0.032, 48),
    new THREE.MeshBasicMaterial({ color: 0x3fbf7f, side: THREE.DoubleSide }));
  const dot = new THREE.Mesh(new THREE.CircleGeometry(0.004, 16), new THREE.MeshBasicMaterial({ color: 0x3fbf7f }));
  g.add(ring, dot);
  g.visible = false;
  scene.add(g);
  world.placeMarker = g;
}

function buildTrail() {
  const n = 600;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  geo.setDrawRange(0, 0);
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xffd166, transparent: true, opacity: 0.7 }));
  line.frustumCulled = false;
  line.userData = { n, count: 0, last: null };
  scene.add(line);
  world.trail = line;
}

function pushTrail(p) {
  const t = world.trail;
  if (!t) return;
  const last = t.userData.last;
  if (last && Math.hypot(p[0] - last[0], p[1] - last[1], p[2] - last[2]) < 0.002) return;
  const arr = t.geometry.attributes.position.array;
  if (t.userData.count === t.userData.n) { arr.copyWithin(0, 3); t.userData.count -= 1; }
  arr.set(p, t.userData.count * 3);
  t.userData.count += 1;
  t.userData.last = p;
  t.geometry.setDrawRange(0, t.userData.count);
  t.geometry.attributes.position.needsUpdate = true;
}

let planKey = '';
function showPlan(plan) {
  const key = plan ? JSON.stringify(plan.path) : '';
  if (key === planKey) return;
  planKey = key;
  if (world.path) { scene.remove(world.path); world.path.geometry.dispose(); world.path = null; }
  world.waypoints.clear();
  if (!plan) return;
  const geo = new THREE.BufferGeometry().setFromPoints(plan.path.map((p) => new THREE.Vector3(...p)));
  world.path = new THREE.Line(geo, new THREE.LineDashedMaterial({ color: 0xffd166, dashSize: 0.01, gapSize: 0.006 }));
  world.path.computeLineDistances();
  scene.add(world.path);
  for (const m of plan.moves) {
    const color = m.then === 'close' ? 0xef5350 : m.then === 'open' ? 0x3fbf7f : 0xffd166;
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.007, 12, 8), new THREE.MeshBasicMaterial({ color }));
    s.position.set(...m.xyz);
    world.waypoints.add(s);
  }
}

// tool target that can be dragged in the view (TransformControls)
const target = new THREE.Group();
target.add(new THREE.AxesHelper(0.07));
target.visible = false;
scene.add(target);
const gizmo = new TransformControls(camera, renderer.domElement);
gizmo.setSpace('local');
gizmo.size = 0.8;
gizmo.addEventListener('dragging-changed', (e) => { orbit.enabled = !e.value && !followCamera; });
gizmo.addEventListener('objectChange', () => throttle('target', 120, previewTarget));
scene.add(gizmo);

function buildScene() {
  buildTable(config.table_z ?? 0);
  robot = buildRobot(false);
  ghost = buildRobot(true);
  ghost.root.visible = false;
  ghost.camera.visible = false;   // it would sit in front of the wrist-camera view
  buildPlaceMarker();
  buildTrail();
  sceneReady = true;
}

// ============================================================
// SERVER LINK
// ============================================================

function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'hello') onHello(msg);
    else if (msg.type === 'state') onState(msg);
    else if (msg.type === 'log') onLog(msg.entries);
    else if (msg.type === 'reply') {
      const resolve = pending.get(msg.id);
      pending.delete(msg.id);
      if (resolve) resolve(msg);
    }
  };
  ws.onclose = () => {
    setBadge($('source'), 'disconnected', 'm-bad');
    for (const resolve of pending.values()) resolve({ ok: false, error: 'disconnected' });
    pending.clear();
    setTimeout(connect, 1000);
  };
}

function send(cmd, args = {}) {
  return new Promise((resolve) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) { resolve({ ok: false, error: 'not connected' }); return; }
    const id = ++reqId;
    pending.set(id, resolve);
    ws.send(JSON.stringify({ cmd, id, ...args }));
  });
}

async function act(cmd, args) {
  const r = await send(cmd, args);
  if (!r.ok) toast(r.error);
  return r;
}

const timers = {};
function throttle(key, ms, fn) {
  if (timers[key]) { timers[key].again = true; return; }
  fn();
  timers[key] = { again: false };
  setTimeout(() => {
    const again = timers[key].again;
    delete timers[key];
    if (again) throttle(key, ms, fn);
  }, ms);
}

function onHello(msg) {
  model = msg.model;
  config = msg.config;
  if (!sceneReady) {
    buildScene();
    buildJointRows();
  }
  setBadge($('source'), config.sim ? 'SIMULATION' : `ROBOT ${config.robot}`, config.sim ? 'sim' : 'live');
  $('speed').max = config.max_speed;
  $('detect').hidden = !config.detect;
  document.title = `Nova 2 Digital Twin · ${config.sim ? 'simulation' : config.robot}`;
}

// ============================================================
// STATE -> SCENE AND PANEL
// ============================================================

const MODE_CLASS = { 5: 'm-idle', 6: 'm-drag', 7: 'm-run', 8: 'm-run', 9: 'm-bad', 11: 'm-bad', 10: 'm-bad' };

function setBadge(el, text, cls) {
  el.textContent = text;
  el.className = `badge ${cls || ''}`;
}

function onState(s) {
  state = s;
  if (!sceneReady) return;
  const r = s.robot;
  if (r.joints) {
    robot.setJoints(r.joints);
    robot.setGrip(s.gripper.position);
    pushTrail([r.tool[0][3], r.tool[1][3], r.tool[2][3]]);
  }
  setBadge($('mode'), r.connected ? r.mode_name : 'no feedback', r.connected ? MODE_CLASS[r.mode] : 'm-bad');
  $('link').textContent = r.connected ? `feedback ${r.age_ms} ms` : '';
  $('arm').checked = s.armed;
  $('armed-banner').hidden = !s.armed;
  $('task').textContent = s.task ? `${s.task}${s.step ? ' — ' + s.step : ''}` : 'idle';
  $('robot-state').textContent = r.connected ? `${r.mode_name} (mode ${r.mode})${r.drag ? ', drag' : ''}` : 'no feedback';
  $('model-check').textContent = s.model_error_mm == null ? '—' : `${s.model_error_mm.toFixed(2)} mm`;
  if (document.activeElement !== $('speed')) { $('speed').value = s.speed; $('speed-out').textContent = `${s.speed} %`; }

  // ghost: a preview, else the planned grasp
  let ghostQ = s.ghost;
  if (!ghostQ && s.plan) ghostQ = (s.plan.moves.find((m) => m.then === 'close') || {}).q;
  ghost.root.visible = !!ghostQ;
  if (ghostQ) ghost.setJoints(ghostQ);
  ghost.setGrip(s.gripper.position);
  showPlan(s.plan);

  // object and place
  if (s.object) {
    if (!world.objectMesh || world.objectMesh.userData.diameter !== s.object.diameter) buildObject(s.object.diameter);
    world.objectMesh.visible = true;
    world.objectMesh.matrix.copy(fromRows(s.object.T));
    const p = s.object.T;
    $('obj-state').textContent = `${mm(p[0][3])}, ${mm(p[1][3])}, ${mm(p[2][3])} mm${s.object.attached ? ' (held)' : ''} · ${s.object.source}`;
  } else {
    if (world.objectMesh) world.objectMesh.visible = false;
    $('obj-state').textContent = '— (Set object, then click the table)';
  }
  world.placeMarker.visible = !!s.place;
  if (s.place) {
    world.placeMarker.position.set(s.place[0], s.place[1], (s.table_z ?? 0) + 0.0012);
    $('place-state').textContent = `${mm(s.place[0])}, ${mm(s.place[1])} mm`;
  } else {
    $('place-state').textContent = '— (pick only)';
  }
  $('pick').textContent = s.place ? 'Pick & place' : 'Pick';
  $('grip-state').textContent = `${s.gripper.holding ? 'holding' : s.gripper.position > 200 ? 'closed' : s.gripper.position > 0 ? 'closed on something' : 'open'} (${s.gripper.position}/255)`;

  // panel numbers
  if (r.joints) {
    r.joints.forEach((a, i) => {
      const row = jointRows[i];
      row.actual.textContent = `actual ${a.toFixed(2)}°`;
      if (!edited[i]) { targets[i] = a; row.range.value = a; row.num.value = a.toFixed(2); }
    });
    const T = r.tool;
    $('tool-xyz').textContent = `${mm(T[0][3])}, ${mm(T[1][3])}, ${mm(T[2][3])} mm`;
    $('tcp-xyz').textContent = r.tcp ? `${r.tcp[0].toFixed(1)}, ${r.tcp[1].toFixed(1)}, ${r.tcp[2].toFixed(1)} mm` : '—';
  }
  $('drag').textContent = r.mode === 6 ? 'End drag' : 'Drag mode';
  refreshButtons();
}

function refreshButtons() {
  const s = state;
  const ok = s && s.robot.connected;
  const busy = !!(s && s.task);
  const canMove = ok && s.armed && !busy && s.robot.mode === 5;
  for (const id of ['joints-move', 'home', 'grip-open', 'grip-close', 'pick', 'target-joint', 'target-line']) {
    $(id).disabled = !canMove || ((id === 'target-joint' || id === 'target-line') && !target.visible);
  }
  document.querySelectorAll('[data-jog]').forEach((b) => { b.disabled = !canMove; });
  jointRows.forEach((row) => { row.minus.disabled = !canMove; row.plus.disabled = !canMove; });
  $('detect').disabled = !canMove;
  for (const id of ['enable', 'disable', 'clear-error', 'drag']) $(id).disabled = !ok || busy;
  $('arm').disabled = !ok;
}

function onLog(entries) {
  const ol = $('log');
  for (const e of entries) {
    const li = document.createElement('li');
    li.className = e.level;
    const t = new Date(e.t * 1000);
    li.innerHTML = `<time>${t.toLocaleTimeString()}</time>`;
    li.append(document.createTextNode(e.text));
    ol.append(li);
    if (e.level === 'error') toast(e.text);
  }
  while (ol.children.length > 200) ol.firstChild.remove();
  ol.scrollTop = ol.scrollHeight;
}

let toastTimer = null;
function toast(text, kind = 'error') {
  const el = $('toast');
  el.textContent = text;
  el.className = kind;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 5000);
}

// ============================================================
// PANEL CONTROLS
// ============================================================

const jointRows = [];
function buildJointRows() {
  const box = $('joints');
  model.servo_limits_deg.forEach(([lo, hi], i) => {
    const row = document.createElement('div');
    row.className = 'joint';
    row.innerHTML = `<span class="name">J${i + 1}</span>
      <input type="range" min="${lo}" max="${hi}" step="0.1">
      <input type="number" min="${lo}" max="${hi}" step="0.5">
      <button title="jog −">−</button><button title="jog +">+</button>
      <span class="actual">actual —</span>`;
    box.append(row);
    const [range, num] = row.querySelectorAll('input');
    const [minus, plus] = row.querySelectorAll('button');
    const r = { el: row, range, num, minus, plus, actual: row.querySelector('.actual') };
    const setTarget = (v) => {
      targets[i] = Math.min(hi, Math.max(lo, Number(v)));
      edited[i] = true;
      row.classList.add('edited');
      range.value = targets[i];
      num.value = targets[i].toFixed(2);
      throttle('joints', 120, previewJoints);
    };
    range.addEventListener('input', () => setTarget(range.value));
    num.addEventListener('change', () => setTarget(num.value));
    minus.addEventListener('click', () => act('jog_joint', { joint: i, delta: -Number($('jog-step').value) }));
    plus.addEventListener('click', () => act('jog_joint', { joint: i, delta: Number($('jog-step').value) }));
    jointRows.push(r);
  });
}

async function previewJoints() {
  const r = await send('preview_joints', { joints: targets.slice() });
  if (!r.ok) toast(r.error, 'info');
}

function resetJoints() {
  edited.fill(false);
  jointRows.forEach((row) => row.el.classList.remove('edited'));
  send('clear_preview');
}

$('joints-reset').addEventListener('click', resetJoints);
$('joints-move').addEventListener('click', async () => {
  const r = await act('move_joints', { joints: targets.slice() });
  if (r.ok) { edited.fill(false); jointRows.forEach((row) => row.el.classList.remove('edited')); }
});
document.querySelectorAll('[data-jog]').forEach((b) => b.addEventListener('click', () => {
  const [axis, sign] = [b.dataset.jog[0], b.dataset.jog[1] === '+' ? 1 : -1];
  act('jog_tool', { axis, delta: sign * Number($('jog-mm').value) });
}));

$('arm').addEventListener('change', (e) => {
  e.preventDefault();
  act('arm', { on: e.target.checked });
});
$('stop').addEventListener('click', () => act('stop'));
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { act('stop'); return; }
  if (target.visible && e.target === document.body) {
    if (e.key === 't') setGizmoMode('translate');
    if (e.key === 'r') setGizmoMode('rotate');
  }
});
$('enable').addEventListener('click', () => act('enable'));
$('disable').addEventListener('click', () => act('disable'));
$('clear-error').addEventListener('click', () => act('clear_error'));
$('drag').addEventListener('click', () => act('drag', { on: !(state && state.robot.mode === 6) }));
$('speed').addEventListener('input', (e) => { $('speed-out').textContent = `${e.target.value} %`; });
$('speed').addEventListener('change', (e) => act('speed', { value: Number(e.target.value) }));
$('home').addEventListener('click', () => act('home'));
$('grip-open').addEventListener('click', () => act('gripper', { action: 'open' }));
$('grip-close').addEventListener('click', () => act('gripper', { action: 'close' }));

// tool target
function setGizmoMode(mode) {
  gizmo.setMode(mode);
  $('target-mode').textContent = mode === 'translate' ? 'Turn' : 'Move';
}
$('target-toggle').addEventListener('click', () => {
  target.visible = !target.visible;
  $('target-toggle').classList.toggle('active', target.visible);
  $('target-mode').disabled = !target.visible;
  if (target.visible && state && state.robot.tool) {
    fromRows(state.robot.tool).decompose(target.position, target.quaternion, new THREE.Vector3());
    gizmo.attach(target);
    setGizmoMode('translate');
  } else {
    gizmo.detach();
    send('clear_preview');
  }
  refreshButtons();
});
$('target-mode').addEventListener('click', () => setGizmoMode(gizmo.getMode() === 'translate' ? 'rotate' : 'translate'));
function targetRows() { target.updateMatrixWorld(); return toRows(target.matrixWorld); }
async function previewTarget() {
  const r = await send('preview_pose', { T: targetRows() });
  if (!r.ok) toast(r.error, 'info');
}
$('target-joint').addEventListener('click', () => act('move_pose', { T: targetRows(), linear: false }));
$('target-line').addEventListener('click', () => act('move_pose', { T: targetRows(), linear: true }));

// pick & place
function setPickMode(mode) {
  pickMode = pickMode === mode ? null : mode;
  $('set-object').classList.toggle('active', pickMode === 'object');
  $('set-place').classList.toggle('active', pickMode === 'place');
  $('pick-hint').hidden = !pickMode;
  $('pick-hint').textContent = pickMode ? `Click on the table to put the ${pickMode === 'object' ? 'object' : 'place spot'}` : '';
}
$('set-object').addEventListener('click', () => setPickMode('object'));
$('set-place').addEventListener('click', () => setPickMode('place'));
$('plan').addEventListener('click', () => act('plan'));
$('pick').addEventListener('click', () => act('pick'));
$('clear-plan').addEventListener('click', () => send('clear_preview'));
$('detect').addEventListener('click', () => act('detect'));

const raycaster = new THREE.Raycaster();
let downAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', async (e) => {
  if (!pickMode || !downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObject(world.table)[0];
  if (!hit) return;
  const xy = [Number(hit.point.x.toFixed(4)), Number(hit.point.y.toFixed(4))];
  const mode = pickMode;
  setPickMode(null);
  if (mode === 'object') await act('set_object', { xy, diameter: Number($('diameter').value) / 1000 });
  else await act('set_place', { xy });
});

// camera views
const VIEWS = {
  iso: { pos: [-1.1, -1.0, 0.85], at: [-0.2, 0.05, 0.15] },
  top: { pos: [-0.25, 0.08, 1.9], at: [-0.25, 0.1, 0] },
  front: { pos: [-1.7, 0.1, 0.35], at: [-0.2, 0.1, 0.2] },
};
document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => {
  followCamera = false;
  $('follow-cam').classList.remove('active');
  orbit.enabled = true;
  const v = VIEWS[b.dataset.view];
  camera.position.set(...v.pos);
  orbit.target.set(...v.at);
  camera.fov = 45;
  camera.updateProjectionMatrix();
  orbit.update();
}));
$('follow-cam').addEventListener('click', () => {
  followCamera = !followCamera;
  $('follow-cam').classList.toggle('active', followCamera);
  orbit.enabled = !followCamera;
  camera.fov = followCamera ? 42 : 45;   // RealSense colour stream, vertical
  camera.updateProjectionMatrix();
  if (!followCamera) { camera.position.set(...VIEWS.iso.pos); orbit.target.set(...VIEWS.iso.at); orbit.update(); }
});

// ============================================================
// RENDER LOOP
// ============================================================

const flipToThree = new THREE.Matrix4().makeRotationX(Math.PI);   // OpenCV camera (z forward, y down) -> three.js
function frame() {
  requestAnimationFrame(frame);
  if (followCamera && robot) {
    robot.root.updateMatrixWorld(true);
    const m = robot.camera.matrixWorld.clone().multiply(flipToThree);
    m.decompose(camera.position, camera.quaternion, new THREE.Vector3());
  } else {
    orbit.update();
  }
  renderer.render(scene, camera);
}
resize();
frame();
connect();

// for tests and the console
window.twin = {
  get state() { return state; },
  get meshesLoaded() { return robotMeshesLoaded; },
  robotJoints: () => (robot ? robot.joints.map((j) => j.rotation.z / DEG) : null),
  send,
};
