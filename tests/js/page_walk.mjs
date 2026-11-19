// The phone page's trained-walk Walker as growbot-box.js builds it (walkerFor): the box's
// calibration with IMU_SIGN [1, 1, 1], gain 1 and turn 0, pinned to z-up, its legs swapped
// from the file's action_map. One JSON line in per tick ({A, G, cmd}, already levelled),
// one line out ({l, r}, wire degrees). A first line {swap: false} leaves the swap out.
import { createInterface } from 'node:readline';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [runnerPath, policyPath, calJson] = process.argv.slice(2);
const { Walker } = await import(pathToFileURL(runnerPath).href);
const cal = Object.assign(JSON.parse(calJson), { IMU_SIGN: [1, 1, 1], gain: 1, turn: 0 });
const w = new Walker(JSON.parse(readFileSync(policyPath, 'utf8')), cal);
w.setFrame('zup');
w.setSwap(!w.swap);
for await (const line of createInterface({ input: process.stdin })) {
  const m = JSON.parse(line);
  if ('swap' in m) { w.setSwap(m.swap); continue; }
  const r = w.step(m.A, m.G, m.cmd);
  process.stdout.write(JSON.stringify({ l: r.l, r: r.r }) + '\n');
}
