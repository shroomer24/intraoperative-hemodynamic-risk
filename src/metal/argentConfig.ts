import { TONE_PARAMS } from 'argentui';
import type { MetalFillProps, MetalProps } from 'argentui';
import { getShaderColorFromString, LiquidMetalShapes, ShaderFitOptions } from '@paper-design/shaders';
import type { ShaderMountUniforms } from '@paper-design/shaders';

export const ARGENT_COMMIT = '6d5d4f4dbcfbd6fe6d7a04762d7bb2da47856f7e';
export const argentConfig = Object.freeze({ tone: 'gunmetal', variant: 'border', frame: 'single', finish: 'surface',
  engine: 'paper', radius: 20, speed: 1.5, metalScale: 1, angle: 30, sheen: true, revealOnHover: true } satisfies MetalProps);
export const fillConfig = Object.freeze({ tone: argentConfig.tone, engine: argentConfig.engine,
  finish: argentConfig.finish, speed: argentConfig.speed, scale: argentConfig.metalScale, angle: argentConfig.angle } satisfies MetalFillProps);
export const DPR_CAP = 1.5;
export function pixelBudget(width: number, height: number): number {
  return Math.max(1, Math.floor(width * height * DPR_CAP * DPR_CAP));
}
// Same shape=none/fit=cover LiquidMetal uniforms as Argent MetalFill and Paper's React adapter.
// No image, URL, mask, alternate tone, finish jitter or native-engine path is used.
export function paperUniforms(): ShaderMountUniforms {
  const preset = TONE_PARAMS.gunmetal;
  return { u_colorBack: getShaderColorFromString(preset.colorBack!), u_colorTint: getShaderColorFromString(preset.colorTint!),
    u_contour: preset.contour, u_distortion: preset.distortion, u_softness: preset.softness, u_repetition: preset.repetition,
    u_shiftRed: preset.shiftRed, u_shiftBlue: preset.shiftBlue, u_angle: fillConfig.angle,
    u_isImage: false, u_shape: LiquidMetalShapes.none, u_fit: ShaderFitOptions.cover, u_scale: fillConfig.scale,
    u_rotation: 0, u_offsetX: 0, u_offsetY: 0, u_originX: .5, u_originY: .5, u_worldWidth: 0, u_worldHeight: 0 };
}
