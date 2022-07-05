import os
import pickle
import blf
from .scanner import Token
# from .parser import MacroType, Compiler
import bpy

add_on_dir = os.path.dirname(
    os.path.realpath(__file__))

font_directory = os.path.join(add_on_dir, 'fonts')
macro_directory = os.path.join(add_on_dir, 'macros')
fonts = {
    'bfont': 0,
    'regular': blf.load(os.path.join(font_directory, 'Anonymous_Pro.ttf')),
    'italic': blf.load(os.path.join(font_directory, 'Anonymous_Pro_I.ttf')),
    'bold': blf.load(os.path.join(font_directory, 'Anonymous_Pro_0.ttf')),
    'bold_italic': blf.load(os.path.join(font_directory, 'Anonymous_Pro_BI.ttf')),
}


class FileData():
    def __init__(self) -> None:
        self.macros: MacroType = {}


file_data = FileData()


@bpy.app.handlers.persistent
def load_macros(dir: str = None, force_update: bool = False) -> list[tuple[str, list[Token]]]:
    global file_data
    if dir is None:
        prefs = bpy.context.preferences.addons['math_formula'].preferences
        dir = prefs.macro_folder
    filenames = os.listdir(dir)
    errors = []
    if not force_update:
        for filename in filenames:
            if filename.startswith('cache'):
                with open(os.path.join(dir, filename), 'rb') as f:
                    file_data.macros = pickle.load(f)
                    f.close()
                    return errors
    else:
        file_data.macros = {}
    for filename in filenames:
        if not filename.startswith('cache'):
            with open(os.path.join(dir, filename), 'r') as f:
                compiler = Compiler()
                success = compiler.compile(f.read(), file_data.macros, 'Any')
                if not success:
                    errors.append((filename, compiler.errors))
                f.close()
    if errors != []:
        return errors
    # Store in cache only if succesful.
    with open(os.path.join(dir, 'cache'), 'w+b') as f:
        pickle.dump(file_data.macros, f)
        f.close()
    return errors


class MF_OT_load_macros(bpy.types.Operator):
    """Load all the macros in the given macro folder.
If there is a cache it will be loaded from there instead,
unless `force_update` is true"""
    bl_idname = "node.mf_load_macros"
    bl_label = "Load macros"
    bl_options = {'REGISTER', 'UNDO'}

    force_update: bpy.props.BoolProperty(
        name="Force Update",
        description="Force reparsing of files, even if there is a cache file",
        default=False
    )

    def execute(self, context: bpy.context):
        global file_data
        prefs = context.preferences.addons['math_formula'].preferences
        errors = load_macros(prefs.macro_folder, self.force_update)
        if errors != []:
            self.report(
                {'ERROR'}, 'Errors when loading macros. See console for more details.')
            for filename, file_errors in errors:
                print(f'Errors in {filename}')
                for file_error in file_errors:
                    print(file_error)
            # Ensure that we don't use wrong macros
            file_data.macros = {}
            return {'CANCELLED'}
        self.report(
            {'INFO'}, f'Succesfully loaded {len(file_data.macros)} macros.')
        return {'FINISHED'}


classes = (
    MF_OT_load_macros,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.app.handlers.load_post.append(load_macros)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
