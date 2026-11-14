// Feed recorded simulator frames through the app's Walker and print its actions.
// Each frame: [accel xyz, gyro xyz, prev0, prev1, cmd_yaw, sin, cos]; the Walker builds
// prev_action and the clock itself, so only the sensor and command parts are passed in.
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [runnerPath, policyPath, casePath] = process.argv.slice(2);
const { Walker } = await import(pathToFileURL(runnerPath).href);
const policy = JSON.parse(readFileSync(policyPath, 'utf8'));
const { frames } = JSON.parse(readFileSync(casePath, 'utf8'));
// IMU_SIGN all +1: the simulator already reports in the policy's body frame.
const w = new Walker(policy, { IMU_SIGN: [1, 1, 1] });
const out = [];
for (const f of frames) {
  const r = w.step({ x: f[0], y: f[1], z: f[2] }, { x: f[3], y: f[4], z: f[5] }, { fwd: 0, yaw: f[8] });
  out.push(r.action);
}
process.stdout.write(JSON.stringify(out));
