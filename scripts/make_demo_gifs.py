import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageSequence
from common import ROOT, dump, force, manifest
from processing import FK


def read_gif(path):
    frames, starts, durations = [], [], []
    elapsed = 0.
    with Image.open(path) as image:
        for frame in ImageSequence.Iterator(image):
            duration = max(10, frame.info.get('duration', 60))
            starts.append(elapsed)
            durations.append(duration)
            elapsed += duration / 1000
            frames.append(frame.convert('RGB'))
    return frames, np.asarray(starts), elapsed


def font(size):
    return ImageFont.truetype(matplotlib.get_data_path() + '/fonts/ttf/DejaVuSans.ttf', size)


def palette(frames):
    count = min(12, len(frames))
    sheet = Image.new('RGB', (400 * count, 150), 'white')
    for i, k in enumerate(np.linspace(0, len(frames) - 1, count).astype(int)):
        sheet.paste(frames[k].resize((400, 150)), (i * 400, 0))
    base = sheet.quantize(colors=240, method=Image.Quantize.MEDIANCUT).getpalette()[:720]
    reserved = [255, 255, 255, 0, 0, 0, 0, 0, 255, 0, 114, 66, 98, 27, 154, 239, 105, 25, 110, 110, 110, 215, 219, 223, 24, 39, 55, 248, 250, 252, 160, 160, 160, 230, 230, 230, 50, 50, 50, 255, 0, 0, 0, 150, 80, 100, 50, 170]
    result = Image.new('P', (1, 1))
    result.putpalette(reserved + base)
    return result


def render(root, source, config, output, preview_dir=None):
    trial = config['trial_id']
    entry = manifest(root).set_index('trial_id').loc[trial]
    frame = pd.read_csv(root / entry['motion_file'])
    prediction = pd.read_csv(root / 'results/predictions' / (trial + '.csv'))
    time = frame.timestamp.to_numpy()
    predicted_time = prediction.time_s.to_numpy()
    labels = force(frame)[1:].astype(np.float32)
    predictions = prediction[[f'pred_imu_{a}_N' for a in 'xyz']].to_numpy()
    fk = FK(str(root / 'kinematics/scene.xml'))
    _, xyz = fk.R_B_G(frame[[f'pos{i}' for i in range(1, 7)]].to_numpy())
    xyz *= 100
    video, video_starts, video_duration = read_gif(source)
    data_knots = np.asarray(config['data_time_s'])
    video_knots = np.asarray(config['video_time_s'])
    if not np.all(np.diff(data_knots) > 0) or not np.all(np.diff(video_knots) > 0):
        raise ValueError('Alignment anchors must be strictly increasing')
    if video_knots[0] < 0 or video_knots[-1] >= video_duration:
        raise ValueError('Alignment is outside the source GIF')
    duration = min(time[-1], data_knots[-1])
    times = np.r_[np.arange(0., duration, .08), duration]
    style = entry['motion_type']
    name = 'Lateral U' if style == 'lateral_u' else 'Fore-aft S'
    color = '#007242' if style == 'lateral_u' else '#621b9a'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure = Figure(figsize=(10.72, 4.8), dpi=100, facecolor='white')
    canvas = FigureCanvasAgg(figure)
    path_axis = figure.add_axes([.025, .11, .44, .82], projection='3d')
    path_axis.view_init(elev=23, azim=-57)
    path_axis.plot(*xyz.T, color='#d7dbdf', lw=1.7)
    path_axis.scatter(*xyz[0], s=28, facecolor='white', edgecolor=color, linewidth=1.5, depthshade=False)
    tail, = path_axis.plot([], [], [], color=color, lw=2.3)
    dot, = path_axis.plot([], [], [], marker='o', markersize=7.5, color='#ef6919', markeredgecolor='white', markeredgewidth=1.)
    for axis, label, values in zip(['x', 'y', 'z'], ['X [cm]', 'Y [cm]', 'Z [cm]'], xyz.T):
        span = max(8., np.ptp(values))
        pad = max(2., span * .13)
        getattr(path_axis, 'set_' + axis + 'lim')(values.min() - pad, values.max() + pad)
        getattr(path_axis, 'set_' + axis + 'label')(label, labelpad=6)
        getattr(path_axis, axis + 'axis').set_major_locator(plt.MaxNLocator(4))
    path_axis.set_box_aspect((1, 1, .85))
    path_axis.tick_params(labelsize=9, pad=0)
    for axis in [path_axis.xaxis, path_axis.yaxis, path_axis.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('#e1e4e8')
        axis._axinfo['grid']['color'] = (.86, .88, .90, .65)
    force_lines, model_lines, cursors, label_dots, model_dots = [], [], [], [], []
    axes = []
    for i, axis in enumerate('xyz'):
        ax = figure.add_axes([.565, .685 - i * .287, .42, .215])
        combined = np.r_[labels[:, i], predictions[:, i]]
        spread = max(np.ptp(combined), .1)
        ax.set_ylim(combined.min() - spread * .14, combined.max() + spread * .14)
        ax.set_xlim(0, duration)
        ax.set_ylabel('$F_' + axis + '$ [N]', fontsize=11, labelpad=8)
        ax.set_xticks(np.arange(0, duration + .01, 1))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(color='#e3e6e9', lw=.7)
        ax.tick_params(labelsize=9)
        line, = ax.plot([], [], color='#000000', lw=1.6, label='IMU label')
        model, = ax.plot([], [], color='#0000ff', lw=1.5, label='M-imu')
        cursor = ax.axvline(0, color='#9099a4', ls=':', lw=.9)
        label_dot, = ax.plot([], [], 'o', color='#000000', markersize=4)
        model_dot, = ax.plot([], [], 'o', color='#0000ff', markersize=4)
        force_lines.append(line)
        model_lines.append(model)
        cursors.append(cursor)
        label_dots.append(label_dot)
        model_dots.append(model_dot)
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Data time [s]', fontsize=11, labelpad=4)
        axes.append(ax)
    axes[0].legend(loc='lower right', bbox_to_anchor=(1, 1.10), ncol=2, frameon=False, fontsize=10, handlelength=2)
    rendered = []
    mapping = []
    title_font, small_font, header_font = font(19), font(14), font(18)
    for i, t in enumerate(times):
        video_time = float(np.interp(t, data_knots, video_knots))
        video_index = int(np.clip(np.searchsorted(video_starts, video_time, side='right') - 1, 0, len(video) - 1))
        index = max(1, np.searchsorted(time, t, side='right'))
        path_t = xyz[:index]
        tail.set_data(path_t[:, 0], path_t[:, 1])
        tail.set_3d_properties(path_t[:, 2])
        point = np.array([np.interp(t, time, xyz[:, j]) for j in range(3)])
        dot.set_data([point[0]], [point[1]])
        dot.set_3d_properties([point[2]])
        count = max(1, np.searchsorted(predicted_time, t, side='right'))
        for j in range(3):
            force_lines[j].set_data(predicted_time[:count], labels[:count, j])
            model_lines[j].set_data(predicted_time[:count], predictions[:count, j])
            cursors[j].set_xdata([t, t])
            label_dots[j].set_data([t], [np.interp(t, predicted_time, labels[:, j])])
            model_dots[j].set_data([t], [np.interp(t, predicted_time, predictions[:, j])])
        canvas.draw()
        plots = Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())
        image = Image.new('RGB', (1600, 600), 'white')
        image.paste(plots, (520, 95))
        image.paste(video[video_index].resize((512, 384), Image.Resampling.LANCZOS), (0, 143))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1600, 45), fill='#182737')
        draw.text((18, 11), f'{name}  |  Trial 07  |  Object 270 g + IMU 16 g', font=title_font, fill='white')
        draw.text((1578, 12), trial, font=header_font, fill='white', anchor='ra')
        for x, label in [(256, 'Robot video'), (760, 'End-effector trajectory'), (1290, 'External force')]:
            draw.text((x, 68), label, font=header_font, fill='#182737', anchor='mm')
        draw.line((520, 99, 520, 571), fill='#e0e4e8', width=1)
        draw.line((1041, 99, 1041, 571), fill='#e0e4e8', width=1)
        draw.text((256, 551), f't = {t:4.2f} s', font=header_font, fill='#182737', anchor='mm')
        draw.text((760, 571), 'Robot base frame', font=small_font, fill='#59636e', anchor='mm')
        rendered.append(image)
        mapping.append(dict(output_frame=i, data_time_s=float(t), source_video_time_s=video_time, source_frame=video_index))
    lookup = palette(rendered)
    indexed = [image.quantize(palette=lookup, dither=Image.Dither.NONE) for image in rendered]
    durations = [80] * (len(indexed) - 1) + [600]
    output.mkdir(parents=True, exist_ok=True)
    target = output / config['output_file']
    indexed[0].save(target, save_all=True, append_images=indexed[1:], duration=durations,
                    loop=0, disposal=1, optimize=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        for n in [0, len(rendered) // 4, len(rendered) // 2, len(rendered) * 3 // 4, len(rendered) - 1]:
            rendered[n].save(preview_dir / f'{style}_{n:03d}.png')
    with Image.open(target) as check:
        actual_frames = check.n_frames
        actual_duration = sum(frame.info.get('duration', 0) for frame in ImageSequence.Iterator(check)) / 1000
    return dict(trial_id=trial, file=target.name, width=1600, height=600, generated_frames=len(rendered),
                gif_frames=actual_frames, playback_duration_s=actual_duration, data_duration_s=float(duration),
                bytes=target.stat().st_size, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                alignment=config, frame_mapping=mapping)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--u-video', type=Path, required=True)
    parser.add_argument('--s-video', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'media')
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    alignment = json.loads((args.root / 'media/alignment.json').read_text())
    report = []
    for style, source in [('lateral_u', args.u_video), ('fore_aft_s', args.s_video)]:
        result = render(args.root, source, alignment[style], args.output, args.preview_dir)
        report.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in ['frame_mapping', 'alignment']}, indent=2), flush=True)
    dump(args.output / 'gif_metadata.json', report)


if __name__ == '__main__':
    main()
