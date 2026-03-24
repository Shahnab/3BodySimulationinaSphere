import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Icosahedron, Environment } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import { Atom, RotateCcw } from 'lucide-react';

const MAX_TRAIL_POINTS = 3000;
const GLOBE_RADIUS = 12;
const MARBLE_RADIUS = 0.3;

// ── Sync-mode constants ───────────────────────────────────────────────────────
// Figure-8 choreography (Chenciner & Montgomery, 2000)
// Standard ICs use G=1, m=1. Scaling positions & velocities by SCALE=4
// requires G_eff = G * SCALE³ = 64  (dimensional analysis: a ∝ G/r², scale r→4r)
const SYNC_SCALE = 4;
const SYNC_G_BASE = 64; // effective G for all sync modes

// Lagrange equilateral triangle: 3 equal-mass bodies at vertices,
// Keplerian orbit with ω = √(3G/a³), circumradius R = a/√3
const LAGRANGE_SIDE = 8; // side length of equilateral triangle

// ── Šuvakov & Dmitrašinović (2013) — 13 new periodic families ─────────────────
// All use equal masses m=1 and G=1 in canonical units.
// Symmetric IC form: r₁=(−x₀,0,0), r₂=(x₀,0,0), r₃=(0,0,0)
//                   v₁=v₂=(vx,vy,0), v₃=(−2vx,−2vy,0)
// Scale S: multiply positions and velocities by S; G_eff = S³
const ORBIT_FAMILIES = [
  // ID             Display name       Family class   x₀        vx         vy        S
  { id: 'butterfly1',  name: 'Butterfly I',    cls: 'I.A.2',    x0: 0.30589, vx: 0.39721, vy: 0.17521, S: 4 },
  { id: 'butterfly2',  name: 'Butterfly II',   cls: 'I.A.4',    x0: 0.39390, vx: 0.43028, vy: 0.12742, S: 4 },
  { id: 'butterfly3',  name: 'Butterfly III',  cls: 'II.C.1',   x0: 0.25076, vx: 0.43458, vy: 0.20950, S: 4 },
  { id: 'butterfly4',  name: 'Butterfly IV',   cls: 'III.A.1',  x0: 0.35021, vx: 0.38038, vy: 0.21062, S: 4 },
  { id: 'bumblebee',   name: 'Bumblebee',      cls: 'I.A.3',    x0: 0.11279, vx: 0.54451, vy: 0.47514, S: 8 },
  { id: 'moth1',       name: 'Moth I',         cls: 'II.A.1',   x0: 0.46262, vx: 0.36710, vy: 0.10383, S: 4 },
  { id: 'moth2',       name: 'Moth II',        cls: 'II.A.2',   x0: 0.49414, vx: 0.37901, vy: 0.09019, S: 4 },
  { id: 'moth3',       name: 'Moth III',       cls: 'II.B.1',   x0: 0.38009, vx: 0.38053, vy: 0.12969, S: 4 },
  { id: 'dragonfly',   name: 'Dragonfly',      cls: 'III.B.1',  x0: 0.07880, vx: 0.59509, vy: 0.32770, S: 8 },
  { id: 'yarn',        name: 'Yarn',           cls: 'III.C.1',  x0: 0.55932, vx: 0.31361, vy: 0.16374, S: 4 },
  { id: 'yinyang1a',   name: 'Yin-yang Ia',    cls: 'III.D.1',  x0: 0.51270, vx: 0.30541, vy: 0.16974, S: 4 },
  { id: 'yinyang1b',   name: 'Yin-yang Ib',    cls: 'IV.A.1',   x0: 0.54090, vx: 0.30681, vy: 0.16754, S: 4 },
  { id: 'yinyang2a',   name: 'Yin-yang IIa',   cls: 'IV.B.1',   x0: 0.44948, vx: 0.32677, vy: 0.21537, S: 4 },
] as const;

type OrbitId = typeof ORBIT_FAMILIES[number]['id'];

// Helper: find orbit definition by id, or null
const findOrbit = (id: string) => ORBIT_FAMILIES.find(o => o.id === id) ?? null;

// Custom shader for a prominent, glowing spherical boundary without city reflections
const GlobeBoundaryMaterial = new THREE.ShaderMaterial({
  uniforms: {
    color: { value: new THREE.Color('#3a4b5c') },
  },
  vertexShader: `
    varying vec3 vNormal;
    void main() {
      vNormal = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform vec3 color;
    varying vec3 vNormal;
    void main() {
      // Fresnel effect: brighter at the glancing angles
      float rim = pow(1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), 3.0);
      float alpha = rim * 0.5 + 0.05;
      gl_FragColor = vec4(color, alpha);
    }
  `,
  transparent: true,
  blending: THREE.AdditiveBlending,
  side: THREE.BackSide,
  depthWrite: false,
});

// Cyan, Magenta, Yellow for high visibility
const BODY_COLORS = ['#00ffff', '#ff00ff', '#ffff00'];

type SyncMode = 'chaos' | 'figure8' | 'lagrange' | OrbitId;

const getInitialState = (syncMode: SyncMode = 'chaos') => {
  // ── Figure-8 choreography ───────────────────────────────────────────────────
  if (syncMode === 'figure8') {
    const S = SYNC_SCALE;
    return [
      { pos: new THREE.Vector3(-0.97000436*S,  0.24308753*S, 0),
        vel: new THREE.Vector3( 0.46620368*S,  0.43236573*S, 0), mass: 1.0, color: BODY_COLORS[0] },
      { pos: new THREE.Vector3(0, 0, 0),
        vel: new THREE.Vector3(-0.93240737*S, -0.86473146*S, 0), mass: 1.0, color: BODY_COLORS[1] },
      { pos: new THREE.Vector3( 0.97000436*S, -0.24308753*S, 0),
        vel: new THREE.Vector3( 0.46620368*S,  0.43236573*S, 0), mass: 1.0, color: BODY_COLORS[2] },
    ];
  }

  // ── Lagrange equilateral triangle ───────────────────────────────────────────
  if (syncMode === 'lagrange') {
    const a = LAGRANGE_SIDE;
    const R = a / Math.sqrt(3);
    const v = Math.sqrt(SYNC_G_BASE / a);
    return [0, 1, 2].map((i) => {
      const θ = (2 * Math.PI * i) / 3;
      return {
        pos: new THREE.Vector3(R * Math.cos(θ), R * Math.sin(θ), 0),
        vel: new THREE.Vector3(-v * Math.sin(θ),  v * Math.cos(θ), 0),
        mass: 1.0,
        color: BODY_COLORS[i],
      };
    });
  }

  // ── Šuvakov & Dmitrašinović (2013) periodic families ───────────────────────
  // Symmetric IC: r₁=(-x₀,0), r₂=(x₀,0), r₃=(0,0)
  //               v₁=v₂=(vx,vy), v₃=(-2vx,-2vy)
  // Scaled by S so that G_eff = S³ keeps dimensional consistency.
  const orbit = findOrbit(syncMode);
  if (orbit) {
    const { x0, vx, vy, S } = orbit;
    return [
      { pos: new THREE.Vector3(-x0 * S, 0, 0), vel: new THREE.Vector3( vx * S,  vy * S, 0), mass: 1.0, color: BODY_COLORS[0] },
      { pos: new THREE.Vector3( x0 * S, 0, 0), vel: new THREE.Vector3( vx * S,  vy * S, 0), mass: 1.0, color: BODY_COLORS[1] },
      { pos: new THREE.Vector3(0, 0, 0),        vel: new THREE.Vector3(-2*vx*S, -2*vy*S, 0), mass: 1.0, color: BODY_COLORS[2] },
    ];
  }

  // ── Chaos mode (original random ICs) ───────────────────────────────────────
  const bodies = BODY_COLORS.map((color) => ({
    pos: new THREE.Vector3(
      (Math.random() - 0.5) * 10,
      (Math.random() - 0.5) * 10,
      (Math.random() - 0.5) * 10
    ),
    vel: new THREE.Vector3(
      (Math.random() - 0.5) * 6,
      (Math.random() - 0.5) * 6,
      (Math.random() - 0.5) * 6
    ),
    mass: 0.5 + Math.random() * 1.5,
    color: color
  }));
  const totalMomentum = new THREE.Vector3();
  let totalMass = 0;
  bodies.forEach(b => {
    totalMomentum.add(b.vel.clone().multiplyScalar(b.mass));
    totalMass += b.mass;
  });
  const comVel = totalMomentum.divideScalar(totalMass);
  bodies.forEach(b => b.vel.sub(comVel));
  return bodies;
};

// Compute total mechanical energy  E = KE + PE
const computeTotalEnergy = (
  state: { pos: THREE.Vector3; vel: THREE.Vector3; mass: number }[],
  G: number
) => {
  let E = 0;
  for (let i = 0; i < 3; i++) {
    E += 0.5 * state[i].mass * state[i].vel.lengthSq();
    for (let j = i + 1; j < 3; j++) {
      const r = state[i].pos.distanceTo(state[j].pos);
      E -= G * state[i].mass * state[j].mass / Math.max(r, 0.05);
    }
  }
  return E;
};

const ThreeBodySimulation = ({
  gravity,
  speed,
  trailLength,
  resetTrigger,
  logsRef,
  syncMode,
}: {
  gravity: number;
  speed: number;
  trailLength: number;
  resetTrigger: number;
  logsRef: React.RefObject<HTMLDivElement | null>;
  syncMode: SyncMode;
}) => {
  const globeRef = useRef<THREE.Mesh>(null);
  const marblesRef = useRef<(THREE.Mesh | null)[]>([]);
  const trailsRef = useRef<(THREE.Line | null)[]>([]);
  const frameCount = useRef(0);
  
  const bodies = useRef(getInitialState());
  const pairStuckTimers = useRef([0, 0, 0]); // timers for pairs: 0-1, 0-2, 1-2
  // Three-body loop detection: sample pairwise distances to measure pattern variance
  const posSnapshotTimer = useRef(0);
  const loopDistHistory = useRef<Array<[number, number, number]>>([]);
  const loopTimer = useRef(0);
  // Sync-mode: reference energy for the dynamic-gravity feedback controller
  const syncRefEnergy = useRef(0);
  const syncGRef = useRef(SYNC_G_BASE); // dynamically adjusted G (sync modes only)

  // Pre-allocate flat arrays for maximum performance
  const trailData = useMemo(() => {
    return BODY_COLORS.map(() => ({
      positions: new Float32Array(MAX_TRAIL_POINTS * 3),
      // Per-vertex speed intensity (0–1): head = current speed, tail = historical speed
      intensities: new Float32Array(MAX_TRAIL_POINTS),
    }));
  }, []);

  // Per-body ShaderMaterial: plasma shimmer trail with speed-reactive white-hot core
  const trailMaterials = useMemo(() => {
    return BODY_COLORS.map(colorHex => {
      const mat = new THREE.ShaderMaterial({
        uniforms: {
          baseColor: { value: new THREE.Color(colorHex) },
          time:      { value: 0 },
        },
        vertexShader: /* glsl */`
          attribute float intensity;
          varying float vIntensity;
          void main() {
            vIntensity = intensity;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: /* glsl */`
          uniform vec3  baseColor;
          uniform float time;
          varying float vIntensity;
          void main() {
            if (vIntensity < 0.004) discard;
            // Non-linear energy falloff: keeps mid-trail bright, snaps tail to black
            float energy = pow(vIntensity, 0.65);
            // Traveling plasma wave: propagates from tail toward head over time
            float shimmer = 0.86 + 0.14 * sin(time * 14.0 + vIntensity * 40.0);
            // White-hot core at head, pure body color at mid, fades at tail
            float whiteness = energy * energy * 0.55;
            vec3 plasma = mix(baseColor * energy, baseColor + vec3(whiteness), whiteness);
            gl_FragColor = vec4(plasma * shimmer, 1.0);
          }
        `,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      mat.toneMapped = false;
      return mat;
    });
  }, []);

  useEffect(() => {
    bodies.current = getInitialState(syncMode);
    pairStuckTimers.current = [0, 0, 0];
    posSnapshotTimer.current = 0;
    loopDistHistory.current = [];
    loopTimer.current = 0;

    // For sync modes, record the reference energy at t=0.
    // Per-orbit G_base = S³ × G_canonical(=1). Figure-8 & Lagrange use SYNC_G_BASE (S=4→64).
    if (syncMode !== 'chaos') {
      const orbit = findOrbit(syncMode);
      const gBase = orbit ? orbit.S ** 3 : SYNC_G_BASE;
      syncGRef.current = gBase;
      syncRefEnergy.current = computeTotalEnergy(bodies.current, gBase);
    }

    // Pre-fill trails with initial positions so they don't streak from origin
    for (let i = 0; i < 3; i++) {
      const pos = bodies.current[i].pos;
      const positions = trailData[i].positions;
      trailData[i].intensities.fill(0); // clear speed history on reset
      for (let j = 0; j < MAX_TRAIL_POINTS; j++) {
        positions[j*3] = pos.x;
        positions[j*3+1] = pos.y;
        positions[j*3+2] = pos.z;
      }
      if (trailsRef.current[i]) {
        trailsRef.current[i].geometry.attributes.position.needsUpdate = true;
      }
    }
  }, [resetTrigger, trailData, syncMode]);

  useFrame((state, delta) => {
    if (delta > 0.1) delta = 0.016; // Cap delta to prevent physics explosions on lag
    
    const dt = delta * speed;
    const steps = 20; // High sub-stepping for accurate orbital mechanics
    const subDt = dt / steps;
    // Sync modes use a smaller softening — bodies follow precise orbits and
    // only come moderately close together, so near-field accuracy matters more.
    const softening = syncMode !== 'chaos' ? 0.15 : 0.5;

    // ── Dynamic gravity control (sync modes only) ─────────────────────────────
    // Compare current total mechanical energy to the reference value seeded at
    // initialisation.  A proportional controller nudges G up/down to restore the
    // orbit without any hard velocity clamping or artificial impulses.
    //   ΔE > 0  →  bodies gained energy (orbit expanding) → raise G to pull back
    //   ΔE < 0  →  bodies lost energy (orbit shrinking)   → lower G to ease off
    let G: number;
    if (syncMode !== 'chaos') {
      const E_now = computeTotalEnergy(bodies.current, syncGRef.current);
      const relErr = (E_now - syncRefEnergy.current) / Math.abs(syncRefEnergy.current || 1);
      // tanh gives a smooth, bounded correction:  ±15 % of G_base at most
      const orbitEntry = findOrbit(syncMode);
      const gBase = orbitEntry ? orbitEntry.S ** 3 : SYNC_G_BASE;
      syncGRef.current = gBase * (1 - Math.tanh(relErr * 3) * 0.15);
      G = syncGRef.current;
    } else {
      G = gravity * 10.0;
    }

    for (let s = 0; s < steps; s++) {
      // Runge-Kutta 4th Order (RK4) Integration
      const currentState = bodies.current.map(b => ({ pos: b.pos.clone(), vel: b.vel.clone(), mass: b.mass }));
      
      const computeDerivatives = (state) => {
        const derivs = state.map(() => ({ dPos: new THREE.Vector3(), dVel: new THREE.Vector3() }));
        
        for (let i = 0; i < 3; i++) {
          derivs[i].dPos.copy(state[i].vel);
        }
        
        for (let i = 0; i < 3; i++) {
          for (let j = i + 1; j < 3; j++) {
            const dir = new THREE.Vector3().subVectors(state[j].pos, state[i].pos);
            const distSq = dir.lengthSq();
            // Softening prevents infinite forces when bodies pass extremely close to each other
            const forceMag = (G * state[i].mass * state[j].mass) / (distSq + softening);
            const force = dir.normalize().multiplyScalar(forceMag);
            
            derivs[i].dVel.add(force.clone().divideScalar(state[i].mass));
            derivs[j].dVel.sub(force.clone().divideScalar(state[j].mass));
          }
        }
        return derivs;
      };

      const k1 = computeDerivatives(currentState);
      
      const stateK2 = currentState.map((b, i) => ({
        pos: b.pos.clone().add(k1[i].dPos.clone().multiplyScalar(subDt * 0.5)),
        vel: b.vel.clone().add(k1[i].dVel.clone().multiplyScalar(subDt * 0.5)),
        mass: b.mass
      }));
      const k2 = computeDerivatives(stateK2);

      const stateK3 = currentState.map((b, i) => ({
        pos: b.pos.clone().add(k2[i].dPos.clone().multiplyScalar(subDt * 0.5)),
        vel: b.vel.clone().add(k2[i].dVel.clone().multiplyScalar(subDt * 0.5)),
        mass: b.mass
      }));
      const k3 = computeDerivatives(stateK3);

      const stateK4 = currentState.map((b, i) => ({
        pos: b.pos.clone().add(k3[i].dPos.clone().multiplyScalar(subDt)),
        vel: b.vel.clone().add(k3[i].dVel.clone().multiplyScalar(subDt)),
        mass: b.mass
      }));
      const k4 = computeDerivatives(stateK4);

      // Apply RK4 weighted sum
      for (let i = 0; i < 3; i++) {
        const body = bodies.current[i];
        
        const dPos = k1[i].dPos.clone()
          .add(k2[i].dPos.clone().multiplyScalar(2))
          .add(k3[i].dPos.clone().multiplyScalar(2))
          .add(k4[i].dPos.clone())
          .multiplyScalar(subDt / 6);
          
        const dVel = k1[i].dVel.clone()
          .add(k2[i].dVel.clone().multiplyScalar(2))
          .add(k3[i].dVel.clone().multiplyScalar(2))
          .add(k4[i].dVel.clone())
          .multiplyScalar(subDt / 6);

        body.pos.add(dPos);
        body.vel.add(dVel);

        // Hard velocity cap — skipped in sync mode (would corrupt the reference orbit)
        if (syncMode === 'chaos') {
          const MAX_SPEED = 20;
          if (body.vel.length() > MAX_SPEED) {
            body.vel.setLength(MAX_SPEED);
          }
        }
      }

      // Sphere-to-sphere collisions — chaos mode only.
      if (syncMode === 'chaos') {
      for (let i = 0; i < 3; i++) {
        for (let j = i + 1; j < 3; j++) {
          const bodyA = bodies.current[i];
          const bodyB = bodies.current[j];
          const distVec = new THREE.Vector3().subVectors(bodyB.pos, bodyA.pos);
          const dist = distVec.length();
          const minDist = MARBLE_RADIUS * 2;
          
          if (dist < minDist && dist > 0.0001) {
            const normal = distVec.clone().normalize();
            const relVel = new THREE.Vector3().subVectors(bodyB.vel, bodyA.vel);
            const velAlongNormal = relVel.dot(normal);
            
            if (velAlongNormal < 0) {
              const restitution = 0.8;
              const impulseMag = -(1 + restitution) * velAlongNormal / (1 / bodyA.mass + 1 / bodyB.mass);
              const impulse = normal.clone().multiplyScalar(impulseMag);
              
              bodyA.vel.sub(impulse.clone().divideScalar(bodyA.mass));
              bodyB.vel.add(impulse.clone().divideScalar(bodyB.mass));
            }
            
            const penetration = minDist - dist;
            const correction = normal.clone().multiplyScalar(penetration / (1/bodyA.mass + 1/bodyB.mass) * 0.5);
            bodyA.pos.sub(correction.clone().divideScalar(bodyA.mass));
            bodyB.pos.add(correction.clone().divideScalar(bodyB.mass));
          }
        }
      }
      } // end collision block (chaos only)

      // Globe boundary — applied to ALL modes so no body ever escapes the globe.
      for (let i = 0; i < 3; i++) {
        const body = bodies.current[i];
        const r = MARBLE_RADIUS * 1.5;
        const distFromCenter = body.pos.length();

        if (distFromCenter > GLOBE_RADIUS - r) {
          const normal = body.pos.clone().normalize();
          const dot = body.vel.dot(normal);
          if (dot > 0) {
            body.vel.sub(normal.clone().multiplyScalar(2 * dot)).multiplyScalar(0.95);
          }
          body.pos.copy(normal.multiplyScalar(GLOBE_RADIUS - r));
        }
      }
    } // end sub-step for loop & loop-detection (chaos mode only) ─────────────────────
    // In sync modes the bodies follow a known predictable orbit, so none of
    // these corrective impulses should fire — they would destabilise the solution.
    if (syncMode === 'chaos') {
    // --- Anti-sticking: Two-body gravitational lock ---
    // If any pair stays within close proximity for > 2 seconds, apply a separating force
    let pairIdx = 0;
    for (let i = 0; i < 3; i++) {
      for (let j = i + 1; j < 3; j++) {
        const dist = bodies.current[i].pos.distanceTo(bodies.current[j].pos);
        if (dist < MARBLE_RADIUS * 10) { // ~3 units — tight gravitational lock threshold
          pairStuckTimers.current[pairIdx] += delta;
        } else {
          pairStuckTimers.current[pairIdx] = Math.max(0, pairStuckTimers.current[pairIdx] - delta);
        }

        if (pairStuckTimers.current[pairIdx] > 2.0) { // 2 seconds of close proximity
          // Push bodies apart along their separation axis + a random perpendicular kick
          const sepDir = new THREE.Vector3().subVectors(bodies.current[i].pos, bodies.current[j].pos).normalize();
          const randVec = new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize();
          // Gram-Schmidt: remove component parallel to sepDir to get a true perpendicular kick
          const perp = randVec.sub(sepDir.clone().multiplyScalar(sepDir.dot(randVec))).normalize();
          const pushDir = sepDir.clone().add(perp.multiplyScalar(0.5)).normalize();

          const boost = 7.0;
          bodies.current[i].vel.add(pushDir.clone().multiplyScalar(boost));
          bodies.current[j].vel.sub(pushDir.clone().multiplyScalar(boost));

          pairStuckTimers.current[pairIdx] = 0; // Reset this pair's timer
          // Clear three-body loop history: the boost spikes pairwise distances,
          // poisoning the stdev window and preventing the loop detector from firing.
          // Give it a fresh start with the new post-boost dynamics.
          loopDistHistory.current = [];
          loopTimer.current = 0;
        }
        pairIdx++;
      }
    }

    // --- Anti-sticking: Three-body repetitive loop ---
    // Sample pairwise distances every 0.1 s to build a history window
    posSnapshotTimer.current += delta;
    if (posSnapshotTimer.current >= 0.1) {
      posSnapshotTimer.current = 0;
      const d01 = bodies.current[0].pos.distanceTo(bodies.current[1].pos);
      const d12 = bodies.current[1].pos.distanceTo(bodies.current[2].pos);
      const d02 = bodies.current[0].pos.distanceTo(bodies.current[2].pos);
      loopDistHistory.current.push([d01, d12, d02]);
      if (loopDistHistory.current.length > 35) loopDistHistory.current.shift();
    }

    // Once we have 30 samples (~3 s), measure variance of each pair's distance
    if (loopDistHistory.current.length >= 30) {
      const hist = loopDistHistory.current;
      const n = hist.length;
      const stdevs = [0, 0, 0];
      for (let pi = 0; pi < 3; pi++) {
        let mean = 0;
        for (let k = 0; k < n; k++) mean += hist[k][pi];
        mean /= n;
        let variance = 0;
        for (let k = 0; k < n; k++) variance += (hist[k][pi] - mean) ** 2;
        stdevs[pi] = Math.sqrt(variance / n);
      }
      // Low variance on all three pair-distances + bodies still moving = repetitive loop
      const totalKE = bodies.current.reduce((acc, b) => acc + 0.5 * b.mass * b.vel.lengthSq(), 0);
      const isLooping = stdevs[0] < 2.0 && stdevs[1] < 2.0 && stdevs[2] < 2.0 && totalKE > 0.5;

      if (isLooping) {
        loopTimer.current += delta;
      } else {
        loopTimer.current = Math.max(0, loopTimer.current - delta * 0.5);
      }

      if (loopTimer.current > 3.0) { // 3 seconds of detected loop
        // Apply a different chaotic impulse to each body to shatter the pattern
        for (let i = 0; i < 3; i++) {
          const chaosDir = new THREE.Vector3(
            Math.random() - 0.5,
            Math.random() - 0.5,
            Math.random() - 0.5
          ).normalize();
          const magnitude = 5.0 + Math.random() * 3.0;
          bodies.current[i].vel.add(chaosDir.multiplyScalar(magnitude));
        }
        loopTimer.current = 0;
        loopDistHistory.current = []; // Clear history so detection restarts fresh
        // Also reset pair timers: the chaos impulse changes all distances,
        // so stale pair-stuck timers would cause an immediate spurious two-body boost.
        pairStuckTimers.current = [0, 0, 0];
      }
    }
    } // end if (syncMode === 'chaos')

    // Update meshes and trails
    for (let i = 0; i < 3; i++) {
      const body = bodies.current[i];
      
      if (marblesRef.current[i]) {
        marblesRef.current[i].position.copy(body.pos);
        
        // Rotate the faceted crystal
        marblesRef.current[i].rotation.x += delta * 0.5 * (i + 1);
        marblesRef.current[i].rotation.y += delta * 0.3 * (i + 1);
      }

      const { positions, intensities } = trailData[i];

      // Update plasma trail time uniform for shimmer animation
      trailMaterials[i].uniforms.time.value = state.clock.elapsedTime;

      // Shift positions right by one vertex
      positions.copyWithin(3, 0, positions.length - 3);
      positions[0] = body.pos.x;
      positions[1] = body.pos.y;
      positions[2] = body.pos.z;

      // Shift intensity history and record current speed-derived brightness
      // Max expected speed ~12 units/s; clamp to [0,1] for the shader
      intensities.copyWithin(1, 0, intensities.length - 1);
      intensities[0] = Math.min(1.0, body.vel.length() / 12.0);

      if (trailsRef.current[i]) {
        trailsRef.current[i].geometry.attributes.position.needsUpdate = true;
        trailsRef.current[i].geometry.attributes.intensity.needsUpdate = true;
        const drawCount = Math.floor((trailLength / 100) * MAX_TRAIL_POINTS);
        trailsRef.current[i].geometry.setDrawRange(0, drawCount);
      }
    }
    
    // Update UI Logs
    if (logsRef.current && frameCount.current % 10 === 0) {
      const b1 = bodies.current[0];
      const b2 = bodies.current[1];
      const b3 = bodies.current[2];

      if (syncMode === 'chaos') {
        // Derive anti-sticking status for display
        const pairNames = ['1-2', '1-3', '2-3'];
        const lockStatus = pairStuckTimers.current
          .map((t, pi) => t > 0.3 ? `${pairNames[pi]}:${t.toFixed(1)}s` : null)
          .filter(Boolean)
          .join(' ') || 'NONE';
        const loopStatus = loopTimer.current > 0.1
          ? `ACTIVE ${loopTimer.current.toFixed(1)}s`
          : 'MONITORING';

        logsRef.current.innerText = `[SYSTEM LOGS]
INTEGRATION: RK4 (RUNGE-KUTTA)
STEP: ${frameCount.current}

ANTI-LOCK: ${lockStatus}
LOOP DETECT: ${loopStatus}

BODY 1 (CYAN)
POS: ${b1.pos.x.toFixed(2)}, ${b1.pos.y.toFixed(2)}, ${b1.pos.z.toFixed(2)}
VEL: ${b1.vel.x.toFixed(2)}, ${b1.vel.y.toFixed(2)}, ${b1.vel.z.toFixed(2)}

BODY 2 (MAGENTA)
POS: ${b2.pos.x.toFixed(2)}, ${b2.pos.y.toFixed(2)}, ${b2.pos.z.toFixed(2)}
VEL: ${b2.vel.x.toFixed(2)}, ${b2.vel.y.toFixed(2)}, ${b2.vel.z.toFixed(2)}

BODY 3 (YELLOW)
POS: ${b3.pos.x.toFixed(2)}, ${b3.pos.y.toFixed(2)}, ${b3.pos.z.toFixed(2)}
VEL: ${b3.vel.x.toFixed(2)}, ${b3.vel.y.toFixed(2)}, ${b3.vel.z.toFixed(2)}`;
      } else {
        const E_now  = computeTotalEnergy(bodies.current, syncGRef.current);
        const E_ref  = syncRefEnergy.current;
        const relErr = ((E_now - E_ref) / Math.abs(E_ref || 1) * 100);
        const orbit  = findOrbit(syncMode);
        const orbitLabel = orbit
          ? `${orbit.name.toUpperCase()} (${orbit.cls})`
          : syncMode === 'figure8' ? 'FIGURE-8 CHOREOGRAPHY'
          : 'LAGRANGE EQUILATERAL';
        const orbitYear = orbit ? 'ŠUVAKOV & DMITRAŠINOVIĆ 2013' : 'CLASSICAL SOLUTION';
        const gBase = orbit ? orbit.S ** 3 : SYNC_G_BASE;

        logsRef.current.innerText = `[SYNC MODE]
ORBIT: ${orbitLabel}
REF:   ${orbitYear}
STEP: ${frameCount.current}

DYNAMIC G CONTROLLER
  G BASE: ${gBase.toFixed(1)}
  G NOW:  ${syncGRef.current.toFixed(3)}
  E REF:  ${E_ref.toFixed(3)}
  E NOW:  ${E_now.toFixed(3)}
  DRIFT:  ${relErr.toFixed(3)} %

BODY 1 (CYAN)
POS: ${b1.pos.x.toFixed(2)}, ${b1.pos.y.toFixed(2)}, ${b1.pos.z.toFixed(2)}
VEL: ${b1.vel.x.toFixed(2)}, ${b1.vel.y.toFixed(2)}, ${b1.vel.z.toFixed(2)}

BODY 2 (MAGENTA)
POS: ${b2.pos.x.toFixed(2)}, ${b2.pos.y.toFixed(2)}, ${b2.pos.z.toFixed(2)}
VEL: ${b2.vel.x.toFixed(2)}, ${b2.vel.y.toFixed(2)}, ${b2.vel.z.toFixed(2)}

BODY 3 (YELLOW)
POS: ${b3.pos.x.toFixed(2)}, ${b3.pos.y.toFixed(2)}, ${b3.pos.z.toFixed(2)}
VEL: ${b3.vel.x.toFixed(2)}, ${b3.vel.y.toFixed(2)}, ${b3.vel.z.toFixed(2)}`;
      }
    }
    frameCount.current++;

    if (globeRef.current) {
      globeRef.current.rotation.y += delta * 0.05;
      globeRef.current.rotation.z += delta * 0.02;
    }
  });

  return (
    <group>
      {/* Prominent Glass Boundary */}
      <Sphere ref={globeRef} args={[GLOBE_RADIUS, 64, 64]}>
        <primitive object={GlobeBoundaryMaterial} attach="material" />
      </Sphere>
      
      {BODY_COLORS.map((color, i) => (
        <group key={i}>
          <group ref={(el) => marblesRef.current[i] = el}>
            {/* Inner glowing core */}
            <Sphere args={[MARBLE_RADIUS * 0.4, 16, 16]}>
              <meshBasicMaterial color={color} toneMapped={false} />
            </Sphere>
            {/* Outer faceted crystal shell */}
            <Icosahedron args={[MARBLE_RADIUS * 1.5, 1]}>
              <meshPhysicalMaterial 
                color={color}
                transmission={0.95}
                opacity={1}
                metalness={0.2}
                roughness={0.05}
                ior={2.0}
                thickness={1.5}
                flatShading={true}
                envMapIntensity={2.0}
                clearcoat={1.0}
                clearcoatRoughness={0.1}
                transparent={true}
              />
            </Icosahedron>
          </group>
          <line ref={(el) => trailsRef.current[i] = el}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={MAX_TRAIL_POINTS}
                array={trailData[i].positions}
                itemSize={3}
                usage={THREE.DynamicDrawUsage}
              />
              <bufferAttribute
                attach="attributes-intensity"
                count={MAX_TRAIL_POINTS}
                array={trailData[i].intensities}
                itemSize={1}
                usage={THREE.DynamicDrawUsage}
              />
            </bufferGeometry>
            <primitive object={trailMaterials[i]} attach="material" />
          </line>
        </group>
      ))}
    </group>
  );
};

export default function App() {
  const [gravity, setGravity] = useState(5.0);
  const [speed, setSpeed] = useState(1.0);
  const [trailLength, setTrailLength] = useState(10); // Percentage 1-100
  const [resetTrigger, setResetTrigger] = useState(0);
  const [syncMode, setSyncMode] = useState<SyncMode>('chaos');
  const logsRef = useRef<HTMLDivElement>(null);

  const handleReset = () => {
    setResetTrigger(prev => prev + 1);
  };

  const cycleSyncMode = () => {
    setSyncMode(prev => {
      const next: SyncMode = prev === 'chaos' ? 'figure8' : prev === 'figure8' ? 'lagrange' : 'chaos';
      setResetTrigger(r => r + 1);
      return next;
    });
  };

  const selectOrbit = (mode: SyncMode) => {
    setSyncMode(mode);
    setResetTrigger(r => r + 1);
  };

  return (
    <div className="w-full h-screen bg-[#020205] text-white overflow-hidden relative font-mono selection:bg-white/20">
      <div className="absolute top-0 left-0 w-full h-full pointer-events-none p-8 flex flex-col justify-between z-10">

        {/* ── TOP BAR ──────────────────────────────────────────── */}
        <div className="flex justify-between items-start">
          {/* Title block */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <div className="w-px h-3 bg-white/20" />
              <span className="text-[9px] tracking-[0.35em] text-gray-500 uppercase">N-Body Gravitational Simulation</span>
            </div>
            <h1 className="text-2xl font-light tracking-[0.25em] uppercase flex items-center gap-3">
              <Atom className="w-5 h-5 text-white/60" />
              3-Body Problem
            </h1>
            <div className="flex items-center gap-3 mt-0.5">
              <div className="h-px w-16 bg-gradient-to-r from-white/30 to-transparent" />
              <span className="text-[8px] tracking-[0.3em] text-white/30">CHAOTIC DYNAMICS ENGINE</span>
            </div>
          </div>

          {/* Status block */}
          <div className="flex flex-col items-end gap-2">
            <div className="flex items-center gap-2 px-3 py-1 rounded-full border border-green-500/30 bg-green-500/5">
              <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
              <span className="text-[9px] tracking-[0.2em] text-green-400">ACTIVE</span>
            </div>
            <div className="text-[8px] tracking-[0.2em] text-gray-600 text-right leading-relaxed">
              <div>SIM MODE: RK4 INTEGRATION</div>
              <div>SUBSTEPS: 20 / FRAME</div>
              {syncMode !== 'chaos' && (() => {
                const o = findOrbit(syncMode);
                return (
                  <div className="text-cyan-400/80">
                    {o ? `${o.name} · ${o.cls}` : syncMode === 'figure8' ? 'FIGURE-8' : 'LAGRANGE L4'}
                  </div>
                );
              })()}
            </div>
          </div>
        </div>

        {/* ── LEFT SIDE PANEL ───────────────────────────────────── */}
        <div className="absolute top-1/2 left-8 -translate-y-1/2 flex flex-col gap-5 w-[155px]">

          {/* Section: Integrator */}
          <div className="border-l border-white/10 pl-3">
            <div className="text-[7px] tracking-[0.35em] text-gray-500 mb-2 uppercase">Integrator</div>
            <div className="flex flex-col gap-1">
              {[
                ['METHOD',    'RK4 (4TH ORDER)'],
                ['SUBSTEPS',  '20 / FRAME'],
                ['SOFTENING', 'ε = 0.50'],
                ['BOUNDARY',  'SPHERE R=12'],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between items-baseline gap-2">
                  <span className="text-[7px] tracking-widest text-gray-600">{k}</span>
                  <span className="text-[8px] tracking-wider text-white/70">{v}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Divider */}
          <div className="h-px bg-gradient-to-r from-white/10 to-transparent" />

          {/* Section: Bodies */}
          <div className="border-l border-white/10 pl-3">
            <div className="text-[7px] tracking-[0.35em] text-gray-500 mb-2 uppercase">Bodies</div>
            {[
              { label: 'BODY 1', color: '#00ffff' },
              { label: 'BODY 2', color: '#ff00ff' },
              { label: 'BODY 3', color: '#ffff00' },
            ].map(({ label, color }) => (
              <div key={label} className="flex items-center gap-2 mb-1.5">
                <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}` }} />
                <span className="text-[8px] tracking-widest text-white/60">{label}</span>
                <div className="ml-auto h-px flex-1 bg-white/5" />
                <span className="text-[7px] text-white/30">MASS VAR</span>
              </div>
            ))}
          </div>

          {/* Section: Anti-sticking / Sync */}
          <div className="border-l border-white/10 pl-3">
            <div className="text-[7px] tracking-[0.35em] text-gray-500 mb-2 uppercase">Stability</div>
            {syncMode === 'chaos' ? (
              <>
                {[
                  ['LOCK TRIGGER', '2.0 s'],
                  ['LOOP WINDOW',  '3.0 s'],
                  ['CHAOS BOOST',  '5–9 U/S'],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between items-baseline gap-2 mb-1">
                    <span className="text-[7px] tracking-widest text-gray-600">{k}</span>
                    <span className="text-[8px] tracking-wider text-white/50">{v}</span>
                  </div>
                ))}
              </>
            ) : (() => {
                const o = findOrbit(syncMode);
                const gBase = o ? o.S ** 3 : SYNC_G_BASE;
                const scaleVal = o ? o.S : SYNC_SCALE;
                return (
                  <>
                    {[
                      ['CONTROLLER',   'ENERGY-PD'],
                      ['GAIN',         'tanh × 0.15'],
                      ['SOFTENING',    'ε = 0.15'],
                      ['SCALE S',      `${scaleVal}×`],
                      ['G BASE',       `${gBase}`],
                      ...(o ? [['CLASS', o.cls]] : []),
                    ].map(([k, v]) => (
                      <div key={k} className="flex justify-between items-baseline gap-2 mb-1">
                        <span className="text-[7px] tracking-widest text-gray-600">{k}</span>
                        <span className="text-[8px] tracking-wider text-cyan-400/70">{v}</span>
                      </div>
                    ))}
                  </>
                );
              })()
            }
          </div>
        </div>

        {/* ── RIGHT SIDE PANEL ──────────────────────────────────── */}
        <div className="absolute top-1/2 right-8 -translate-y-1/2 flex flex-col gap-5 w-[145px] items-end">

          {/* Section: Simulation params */}
          <div className="border-r border-white/10 pr-3 text-right w-full">
            <div className="text-[7px] tracking-[0.35em] text-gray-500 mb-2 uppercase">Parameters</div>
            {[
              ['GRAVITY G',    `${gravity.toFixed(1)} ×`],
              ['TIME SCALE',   `${speed.toFixed(1)} ×`],
              ['TRAIL RENDER', `${trailLength}%`],
              ['MASS RANGE',   '0.5 – 2.0 M'],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between items-baseline gap-2 mb-1">
                <span className="text-[7px] tracking-widest text-gray-600">{k}</span>
                <span className="text-[8px] tracking-wider text-white/70">{v}</span>
              </div>
            ))}
          </div>

          {/* Divider */}
          <div className="h-px bg-gradient-to-l from-white/10 to-transparent w-full" />

          {/* Section: Physics model */}
          <div className="border-r border-white/10 pr-3 text-right w-full">
            <div className="text-[7px] tracking-[0.35em] text-gray-500 mb-2 uppercase">Physics Model</div>
            {(() => {
              const o = findOrbit(syncMode);
              const rows = syncMode === 'chaos'
                ? [['LAW','NEWTON 1687'],['COLLISION','ELASTIC e=0.8'],['RESTITUTION','0.95 DAMPING'],['SYMMETRY','COM CORRECTED']]
                : [['LAW','NEWTON 1687'],['CONTROL','DYN-G FEEDBACK'],['COLLISION','DISABLED'],['SOURCE', o ? 'PRL 110 114301' : 'CLASSICAL']];
              return rows.map(([k, v]) => (
                <div key={k} className="flex justify-between items-baseline gap-2 mb-1">
                  <span className="text-[7px] tracking-widest text-gray-600">{k}</span>
                  <span className={`text-[8px] tracking-wider ${syncMode !== 'chaos' ? 'text-cyan-400/60' : 'text-white/50'}`}>{v}</span>
                </div>
              ));
            })()}
          </div>

          {/* Decorative corner mark */}
          <div className="flex items-center gap-1.5 text-white/15">
            <div className="h-px w-8 bg-white/10" />
            <span className="text-[7px] tracking-widest">∞ LOOP</span>
          </div>
        </div>

        {/* ── BOTTOM: CONTROLS + TELEMETRY ─────────────────────── */}
        <div className="flex justify-between items-end gap-6">

          {/* Controls panel */}
          <div className="pointer-events-auto bg-black/60 border border-white/10 rounded-lg backdrop-blur-sm overflow-hidden">
            {/* Panel header */}
            <div className="flex items-center justify-between px-4 py-2 border-b border-white/8 bg-white/3">
              <span className="text-[8px] tracking-[0.3em] text-gray-500 uppercase">Simulation Controls</span>
              <div className="flex items-center gap-2">
                {/* Orbit selector dropdown */}
                <select
                  value={syncMode}
                  onChange={(e) => selectOrbit(e.target.value as SyncMode)}
                  className={`px-2 py-1 rounded border text-[8px] tracking-widest uppercase bg-black/80 outline-none cursor-pointer transition-all
                    ${syncMode !== 'chaos'
                      ? 'border-cyan-500/50 text-cyan-300'
                      : 'border-white/15 text-white/60 hover:border-white/30 hover:text-white/80'}`}
                >
                  <optgroup label="─── Classic ──────────────">
                    <option value="chaos">⟳ Chaos</option>
                    <option value="figure8">∞ Figure-8</option>
                    <option value="lagrange">△ Lagrange</option>
                  </optgroup>
                  <optgroup label="─── Šuvakov & Dmitrašinović (2013) ───">
                    {ORBIT_FAMILIES.map(o => (
                      <option key={o.id} value={o.id}>{o.name} ({o.cls})</option>
                    ))}
                  </optgroup>
                </select>
                <button
                  onClick={handleReset}
                  className="flex items-center gap-1.5 hover:bg-white/10 text-white/60 hover:text-white px-2.5 py-1 rounded border border-white/10 hover:border-white/20 transition-all text-[8px] tracking-widest uppercase"
                >
                  <RotateCcw className="w-2.5 h-2.5" />
                  Reset
                </button>
              </div>
            </div>

            {/* Sliders */}
            <div className="flex gap-8 px-5 py-4">
              {/* Gravity */}
              <div className="flex flex-col gap-2 w-28">
                <div className="flex justify-between text-[8px] tracking-wider">
                  <span className="text-gray-500">GRAVITY</span>
                  <span className="text-white tabular-nums">{gravity.toFixed(1)} ×</span>
                </div>
                <input type="range" min="1.0" max="15.0" step="0.5" value={gravity}
                  onChange={(e) => setGravity(parseFloat(e.target.value))}
                  className="w-full accent-white h-px bg-white/20 rounded-full appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:shadow-[0_0_6px_rgba(255,255,255,0.6)]" />
                <div className="flex justify-between text-[7px] text-gray-600">
                  <span>1.0</span><span>15.0</span>
                </div>
              </div>

              {/* Speed */}
              <div className="flex flex-col gap-2 w-28">
                <div className="flex justify-between text-[8px] tracking-wider">
                  <span className="text-gray-500">TIME SCALE</span>
                  <span className="text-white tabular-nums">{speed.toFixed(1)} ×</span>
                </div>
                <input type="range" min="0.1" max="3.0" step="0.1" value={speed}
                  onChange={(e) => setSpeed(parseFloat(e.target.value))}
                  className="w-full accent-white h-px bg-white/20 rounded-full appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:shadow-[0_0_6px_rgba(255,255,255,0.6)]" />
                <div className="flex justify-between text-[7px] text-gray-600">
                  <span>0.1×</span><span>3.0×</span>
                </div>
              </div>

              {/* Trail */}
              <div className="flex flex-col gap-2 w-28">
                <div className="flex justify-between text-[8px] tracking-wider">
                  <span className="text-gray-500">TRAIL</span>
                  <span className="text-white tabular-nums">{trailLength}%</span>
                </div>
                <input type="range" min="1" max="100" step="1" value={trailLength}
                  onChange={(e) => setTrailLength(parseInt(e.target.value))}
                  className="w-full accent-white h-px bg-white/20 rounded-full appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:shadow-[0_0_6px_rgba(255,255,255,0.6)]" />
                <div className="flex justify-between text-[7px] text-gray-600">
                  <span>1%</span><span>100%</span>
                </div>
              </div>
            </div>

            {/* Hint bar */}
            <div className="px-5 py-1.5 border-t border-white/5 flex items-center gap-3 text-[7px] tracking-widest text-gray-600">
              <span>DRAG TO ROTATE</span>
              <span className="text-white/10">•</span>
              <span>SCROLL TO ZOOM</span>
              <span className="text-white/10">•</span>
              <span>REAL-TIME PHYSICS</span>
            </div>
          </div>

          {/* Telemetry panel */}
          <div className="flex flex-col items-end gap-0">
            {/* Panel header */}
            <div className="w-full flex items-center justify-between px-3 py-1.5 bg-black/70 border border-b-0 border-green-500/20 rounded-t-lg">
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                <span className="text-[8px] tracking-[0.3em] text-green-500/70 uppercase">Live Telemetry</span>
              </div>
              <span className="text-[7px] text-gray-600 tracking-widest">3 BODIES</span>
            </div>
            {/* Log body */}
            <div
              ref={logsRef}
              className="text-[8.5px] leading-[1.55] tracking-wider text-green-400/80 bg-black/70 px-3 py-2.5 border border-green-500/20 rounded-b-lg backdrop-blur-md whitespace-pre font-mono text-right w-[230px] h-[300px] overflow-hidden"
            >
              INITIALIZING PHYSICS ENGINE...
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-3 text-[8px] tracking-[0.4em] text-white/20 uppercase select-none">
          <div className="h-px w-10 bg-white/10" />
          Concept by Shahnab
          <div className="h-px w-10 bg-white/10" />
        </div>

      </div>
      
      <Canvas camera={{ position: [0, 0, 35], fov: 45 }}>
        <color attach="background" args={['#020205']} />
        <OrbitControls 
            autoRotate 
            autoRotateSpeed={0.5} 
            enablePan={false} 
            enableZoom={true} 
            minDistance={15}
            maxDistance={60}
        />
        <ThreeBodySimulation 
          gravity={gravity} 
          speed={speed} 
          trailLength={trailLength}
          resetTrigger={resetTrigger}
          logsRef={logsRef}
          syncMode={syncMode}
        />
        <Environment preset="city" />
        <EffectComposer disableNormalPass>
          <Bloom mipmapBlur intensity={1.5} luminanceThreshold={0.1} luminanceSmoothing={0.9} />
        </EffectComposer>
      </Canvas>
    </div>
  );
}
