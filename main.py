import flet as ft
from flet import (
    Page, 
    Column, 
    Row, 
    Container, 
    TextField, 
    ElevatedButton, 
    IconButton, 
    Icons,
    Image,
    FilePicker,
    FilePickerUploadEvent,
    CrossAxisAlignment,
    MainAxisAlignment,
)
import back
from pathlib import Path
from PIL import Image as PILImage

lamp_load_colum = False
num_page = 0
len_directory = 0
scale = 0
index = 0
language = True
color = "#000000"
display = True

font_files  = {
        "Arial": "fonts\ARIAL.TTF",
        "Comic Sans":"fonts\COMIC.TTF",
        "Courier New":"fonts\COUR.TTF",
        "Times New Roman":"fonts\TIMES.TTF",
    }
fonts = list(font_files.keys()) 
current_font = font_files[fonts[0]]

def main(page: Page):
    page.title="Manga translator"
    page.theme_mode=ft.ThemeMode.DARK
    page.padding = ft.padding.only(left = 50, top = 30, bottom = 30)
    #----------------------------------------------------------
    def click_box(e):
        global index, num_page
        index = e.control.data
        delete_button.disabled = False
        confirmation_button.disabled = False
        img_name = back.get_image_name(num_page)
        text_path = "storage/paddleocr_annotation/"+img_name+"/"+str(index)+"/text.txt"
        text=back.read_text_prediction(text_path)
        text_editor.value = text
        page.update()

    def delete_box(e):
        global index, num_page
        img_name = back.get_image_name(num_page)
        yolo_storage_path = "storage/yolo_annotation/"
        img_paddleocr_annotation_path = "storage/paddleocr_annotation/"+img_name
        img_annotation_path=yolo_storage_path + img_name + ".txt"
        back.delete_yolo_annotation(img_annotation_path, index)
        back.delete_paddleocr_annotation(img_paddleocr_annotation_path, index)
        switch_iteration()
    #----------------------------------------------------------
    def switch_disabled():
        global num_page, len_directory
        if num_page == 1:
            button_left.disabled = True
        else:
            button_left.disabled = False
        if num_page == len_directory:
            button_right.disabled = True
        else:
            button_right.disabled = False
        control_panel.value = num_page

    def switch_image():
        global num_page, scale
        storage_path = "storage/images/"
        img_file = storage_path + back.get_image_number(num_page)
        pil_img = PILImage.open(img_file)
        orig_w, orig_h = pil_img.size
        cont_w = page.width * 0.8
        cont_h = page.height * 0.8
        scale = min(cont_w / orig_w, cont_h / orig_h)
        disp_w = orig_w * scale
        disp_h = orig_h * scale
        img = ft.Image(
            src=img_file,
            width=disp_w,
            height=disp_h,
        )
        image_container.content = ft.Stack(
            [img],
            width=disp_w,
            height=disp_h
        )
        page.update()

    def draw_boxes(index,text ,x1, y1, x2, y2):
        global scale
        rect = ft.Container(
            content=ft.Text(text, color=ft.Colors.RED_500, weight=ft.FontWeight.BOLD),
            alignment=ft.Alignment.CENTER,
            left=x1 * scale,
            top=y1 * scale,
            width=(x2 - x1) * scale,
            height=(y2 - y1) * scale,
            bgcolor=ft.Colors.with_opacity(0.8, ft.Colors.BLUE),
            border=ft.Border.all(1, ft.Colors.BLUE),
            data = index,
            on_click=click_box
        )
        image_container.content.controls.append(rect)

    def create_boxes():
        global num_page
        img_name = back.get_image_name(num_page)
        yolo_storage_path = "storage/yolo_annotation/"
        img_annotation_path=yolo_storage_path + img_name + ".txt"
        boxes = back.read_yolo_annotation(img_annotation_path)
        for box in boxes:
            text_path = "storage/paddleocr_annotation/"+img_name+"/"+str(box["index"])+"/text.txt"
            text=back.read_text_prediction(text_path)
            draw_boxes(box["index"],text,box["x_min"],box["y_min"],box["x_max"],box["y_max"])

    def switch_iteration():
        global index, display
        index = 0
        delete_button.disabled=True
        confirmation_button.disabled = True
        text_editor.value =""
        switch_disabled()
        switch_image()
        if display:
            create_boxes()
        page.update()
    #----------------------------------------------------------
    async def get_file (e: ft.Event[ElevatedButton]):
        global language
        path = await ft.FilePicker().pick_files()
        back.download_file(path[0].path, language)
        global num_page, len_directory
        num_page = 1
        len_directory = back.get_len_directory()
        save_button.disabled = False
        display_button.disabled=False
        switch_iteration()

    async def get_directory (e: ft.Event[ElevatedButton]):
        global language
        path = await ft.FilePicker().get_directory_path()
        back.download_directory(path, language)
        global num_page, len_directory
        num_page = 1
        len_directory = back.get_len_directory()
        save_button.disabled = False
        display_button.disabled=False
        switch_iteration()

    upload_file = upload_button = ElevatedButton(
        content=ft.Text("Загрузить файл"),
        icon=Icons.LIST,
        on_click=get_file
    )

    upload_directory = upload_button = ElevatedButton(
        content=ft.Text("Загрузить папку"),
        icon=Icons.FOLDER,
        on_click=get_directory
    )

    load_colum = Column(
        controls=[]
    )
    #----------------------------------------------------------
    async def save_files(e):
        global len_directory, color, current_font
        save_path = await ft.FilePicker().get_directory_path()
        for i in range(len_directory):
            image_path="storage/images/"+str(back.get_image_number(i+1))
            image_name=back.get_image_name(i+1)
            back.apply_paddleocr_annotations(image_path, image_name, save_path, text_color_hex=color, font_path=current_font)

    def file_pick(e: ft.Event[ElevatedButton]):
        global lamp_load_colum
        if lamp_load_colum:
            load_colum.controls = []
            lamp_load_colum=False
        else:
            load_colum.controls = [upload_file, upload_directory]
            lamp_load_colum=True
        page.update()

    upload_button = ElevatedButton(
        content=ft.Text("Загрузить"),
        icon=Icons.UPLOAD,
        on_click=file_pick
    )

    save_button = ElevatedButton(
        content=ft.Text("Сохранить"),
        icon=Icons.SAVE,
        disabled=True,
        on_click=save_files
    )

    top_panel = Row(
        controls=[upload_button, save_button],
    )
    #----------------------------------------------------------
    text_editor = TextField(
        hint_text="Выберите поле",
        bgcolor="white",
        color="black",
        multiline=True, 
        min_lines=10,     
        max_lines=10, 
        width=280
    )

    def confirmation_text(e):
        global index, num_page
        img_name = back.get_image_name(num_page)
        text_path = "storage/paddleocr_annotation/"+img_name+"/"+str(index)+"/text.txt"
        text=text_editor.value
        back.write_text_prediction(text_path, text)
        switch_iteration()
    
    confirmation_button = ElevatedButton(
        content=ft.Text("Подтвердить"),
        icon=Icons.CONFIRMATION_NUM,
        disabled=True,
        on_click=confirmation_text
    )

    tex_panel = Column(
        controls=[text_editor, confirmation_button],
        margin=ft.margin.only(top=100)
    )
    #----------------------------------------------------------
    delete_button = ElevatedButton(
        content=ft.Text("Удалить"),
        icon=Icons.DELETE,
        disabled=True,
        on_click=delete_box
    )

    color_input = ft.TextField(
        label="Цвет",
        value="#000000",
        width=120
    )

    color_preview = ft.Container(
        width=40,
        height=40,
        bgcolor=color_input.value,
        border_radius=6
    )

    def apply_color(e):
        global color
        color_preview.bgcolor = color_input.value
        color=color_input.value
        color_preview.update()

    color_picker_row = ft.Row(
        [
            color_input,
            ft.ElevatedButton("OK", on_click=apply_color, height=40),
            color_preview
        ],
        spacing=10,
        vertical_alignment=ft.CrossAxisAlignment.CENTER
    )
    #----------------------------------------------------------
    def not_display_boxes(e):
        global display
        not_display_button.disabled=True
        display_button.disabled=False
        display = True
        switch_iteration()

    def display_boxes(e):
        global display
        not_display_button.disabled=False
        display_button.disabled=True
        display = False
        switch_iteration()

    display_button = ElevatedButton(
        content=ft.Text("Есть"),
        icon = Icons.REMOVE_RED_EYE,
        disabled=True,
        on_click=display_boxes
    )

    not_display_button = ElevatedButton(
        content=ft.Text("Нету"),
        icon = Icons.REMOVE_RED_EYE_OUTLINED,
        disabled=True,
        on_click=not_display_boxes
    )
    #----------------------------------------------------------
    def swith_rigth_read_button(e):
        global language
        language = False
        rigth_read_button.disabled=True
        left_read_button.disabled=False
        page.update()

    def swith_left_read_button(e):
        global language
        language = True
        rigth_read_button.disabled=False
        left_read_button.disabled=True
        page.update()

    rigth_read_button = ElevatedButton(
        content=ft.Text("С парава"),
        disabled=False,
        on_click=swith_rigth_read_button
    )

    left_read_button = ElevatedButton(
        content=ft.Text("С лева"),
        disabled=True,
        on_click=swith_left_read_button
    )
    #----------------------------------------------------------
    preview_text = ft.Text(
        "Пример",
        size=13,
        font_family=current_font,
    )

    def on_font_change(e):
        global current_font
        selected_name = e.control.value
        current_font = font_files[selected_name]  
        preview_text.font_family = selected_name          
        preview_text.update()


    dropdown = ft.Dropdown(
        label="Шрифт",
        width=220,
        options=[ft.dropdown.Option(name) for name in fonts],
        value=fonts[0],
        on_select=on_font_change
    )

    font_change = ft.Row(
        controls=[dropdown, preview_text],
        alignment=ft.MainAxisAlignment.START,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    edit_panel = Column(
        controls=[
                ft.Text("Перевод"),
                Row(controls=[left_read_button, rigth_read_button],  alignment=MainAxisAlignment.CENTER),
                ft.Text("Показ рамок"),
                Row(controls=[display_button, not_display_button],  alignment=MainAxisAlignment.CENTER),
                ft.Text("Выбор цвета"),
                color_picker_row,
                ft.Text("Выбор шрифта"),
                font_change,
                delete_button
                ],
        horizontal_alignment=CrossAxisAlignment.CENTER
    )
    #----------------------------------------------------------
    def switch_right(e):
        global num_page
        num_page+=1
        switch_iteration()

    def switch_left(e):
        global num_page
        num_page-=1
        switch_iteration()

    button_right = IconButton(
        icon=Icons.ARROW_RIGHT, 
        disabled=True,
        on_click= switch_right
        )
    
    button_left = IconButton(
        icon=Icons.ARROW_LEFT,  
        disabled=True,
        on_click= switch_left
        )
    
    control_panel = ft.Text("0")

    switch_panel = Row(
        controls=[button_left, control_panel, button_right],
        alignment=MainAxisAlignment.CENTER 
    )
    #----------------------------------------------------------
    image_container = Container(
        content=ft.Text("Зона изображения", color=ft.Colors.WHITE),
        expand=True,
        padding=10,
        bgcolor=ft.Colors.GREY_800,
        border_radius=1,
        alignment=ft.Alignment.CENTER,
    )
    #----------------------------------------------------------
    main_container = Row(
        controls=[
            Column(
                controls=[top_panel, load_colum, tex_panel],
                width=300,
            ),
            Column(
                controls=[image_container],
                expand=True,
                horizontal_alignment=CrossAxisAlignment.CENTER,
            ),
            Column(
                controls=[edit_panel, switch_panel],
                width=300,
                alignment=MainAxisAlignment.SPACE_BETWEEN,
                horizontal_alignment=CrossAxisAlignment.CENTER

            ),
        ],
        expand=True,
        vertical_alignment=CrossAxisAlignment.START
    )
    
    page.add(main_container)

ft.app(target=main)