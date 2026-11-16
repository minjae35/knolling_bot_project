"""Camera-only Blender detail views of saved publication scenes.

Uses new output filenames exclusively. Every object transform, vertex, material,
light and world setting is preserved from the source .blend. No model is run.
Run with Blender 5.1: blender -b -t 8 --python scripts/render_detail_views.py
"""
from pathlib import Path
import hashlib
import json
import sys
import bpy

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'renders'
JOBS = [
    ('detail_full_workspace', 'full_after', (-.02, .62, -.01333333333333334, .41333333333333333), (1920,1280)),
    ('detail_row_gap', 'smoke_full_overhead', (.17,.45,.115,.22), (2040,765)),
    ('detail_clearance', 'smoke_partial_overhead', (.13,.31,.162,.24008823529411765), (2040,885)),
]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def scene_fidelity(scene):
    """Values independent of camera/framing and rendering resolution."""
    result = {}
    for ob in sorted(scene.objects, key=lambda v: v.name):
        if ob.type == 'CAMERA':
            continue
        entry = {'type': ob.type, 'matrix': [list(v) for v in ob.matrix_world]}
        if ob.type == 'MESH':
            entry['vertices'] = [list(v.co) for v in ob.data.vertices]
            entry['polygons'] = [list(p.vertices) for p in ob.data.polygons]
            entry['materials'] = []
            for mat in ob.data.materials:
                bsdf=mat.node_tree.nodes.get('Principled BSDF')
                entry['materials'].append({'name':mat.name,
                    'base_color':list(bsdf.inputs['Base Color'].default_value),
                    'roughness':float(bsdf.inputs['Roughness'].default_value)})
        if ob.type == 'LIGHT':
            entry['light']={'type':ob.data.type,'energy':ob.data.energy,
                'size':ob.data.size,'color':list(ob.data.color)}
        result[ob.name]=entry
    bg=scene.world.node_tree.nodes['Background']
    result['_world']={'color':list(bg.inputs['Color'].default_value),
                     'strength':bg.inputs['Strength'].default_value}
    result['_view']={k:getattr(scene.view_settings,k) for k in ['view_transform','look','exposure','gamma']}
    return result

def digest_state(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def main():
    frozen=json.loads((ROOT/'audit/figure6_frozen.json').read_text())
    def preserve():
        assert all(sha(ROOT/n)==h for n,h in frozen['sha256'].items())
        assert frozen['latex_figure_block'] in (ROOT/'appendix.tex').read_text()
    preserve()
    report={'blender_version':bpy.app.version_string,'scope':'Camera-only presentation crops of saved scenes; not new inference or evaluation.',
            'renderer':'Cycles CPU','samples':128,'threads':8,'renders':{}}
    for name,source,extent,resolution in JOBS:
        assert name.startswith('detail_')
        origin=R/(source+'.blend');source_hash=sha(origin)
        bpy.ops.wm.open_mainfile(filepath=str(origin))
        scene=bpy.context.scene
        before=scene_fidelity(scene)
        xmin,xmax,ymin,ymax=extent
        assert abs((xmax-xmin)/(ymax-ymin)-resolution[0]/resolution[1])<1e-10
        camera=scene.camera
        camera.location=((xmin+xmax)/2,(ymin+ymax)/2,1.)
        camera.rotation_euler=(0,0,0)
        camera.data.type='ORTHO';camera.data.sensor_fit='HORIZONTAL'
        camera.data.ortho_scale=xmax-xmin
        scene.render.resolution_x,scene.render.resolution_y=resolution
        scene.render.resolution_percentage=100
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=128
        scene.cycles.seed=20261115;scene.cycles.use_animated_seed=False
        scene.render.threads_mode='FIXED';scene.render.threads=8
        scene.render.filepath=str(R/(name+'.png'))
        bpy.context.view_layer.update()
        assert scene_fidelity(scene)==before
        output=R/(name+'.blend')
        bpy.ops.wm.save_as_mainfile(filepath=str(output),check_existing=False)
        bpy.ops.render.render(write_still=True)
        bpy.ops.wm.open_mainfile(filepath=str(output))
        assert scene_fidelity(bpy.context.scene)==before
        assert sha(origin)==source_hash
        report['renders'][name]={'source_blend':str(origin.relative_to(ROOT)),
            'source_sha256':source_hash,'blend_sha256':sha(output),
            'image_sha256':sha(R/(name+'.png')),'image':name+'.png',
            'world_extent_m':list(extent),'resolution':list(resolution),
            'camera_location':list(bpy.context.scene.camera.location),
            'camera_euler':list(bpy.context.scene.camera.rotation_euler),
            'ortho_scale_m':bpy.context.scene.camera.data.ortho_scale,
            'noncamera_scene_sha256':digest_state(before),
            'noncamera_scene_unchanged':True,'saved_file_reopened_and_checked':True}
        (ROOT/'audit/detail_render_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
        print('DETAIL_RENDER_VERIFIED',name,flush=True)
    preserve()

if __name__=='__main__':
    main()
