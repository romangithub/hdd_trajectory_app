from __future__ import annotations
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from pathlib import Path

def _kv_to_table(d: dict) -> Table:
    data = [["Параметр", "Значение"]]
    for k, v in d.items():
        data.append([str(k), str(v)])
    t = Table(data, colWidths=[70*mm, 100*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    return t

def make_engineer_report(
    filename: str,
    project_name: str,
    section_title: str,
    params: dict | None,
    segments: list | dict | None,
    stations: "pandas.DataFrame | None" = None,
    mud: dict | None = None,
    plot_profile_path: str | None = None,
    plot_plan_path: str | None = None,
    extra_notes: str | None = None,
):
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"<b>{project_name}</b>", styles["Title"]))
    story.append(Paragraph(section_title, styles["h2"]))
    story.append(Spacer(1, 6))

    if params:
        story.append(Paragraph("<b>Исходные параметры</b>", styles["h3"]))
        story.append(_kv_to_table(params))
        story.append(Spacer(1, 6))

    if segments:
        story.append(Paragraph("<b>Сегменты / Этапы</b>", styles["h3"]))
        if isinstance(segments, dict):
            story.append(_kv_to_table(segments))
        else:
            # превратим список словарей в таблицу
            keys = list(segments[0].keys()) if segments else []
            data = [keys] + [[str(row.get(k, "")) for k in keys] for row in segments]
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('FONTSIZE', (0,0), (-1,-1), 8),
            ]))
            story.append(t)
        story.append(Spacer(1, 6))

    if stations is not None:
        story.append(Paragraph("<b>Таблица станций</b>", styles["h3"]))
        # обрежем до первых 40 строк, чтобы не раздуть PDF
        df = stations.head(40).copy()
        data = [list(df.columns)] + df.astype(str).values.tolist()
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTSIZE', (0,0), (-1,-1), 7),
        ]))
        story.append(t)
        story.append(Spacer(1, 6))

    if mud:
        story.append(Paragraph("<b>Mud: ключевые результаты</b>", styles["h3"]))
        story.append(_kv_to_table(mud))
        story.append(Spacer(1, 6))

    for img_path, title in [(plot_profile_path, "Профиль (вертикальная проекция)"),
                            (plot_plan_path, "План (горизонтальная проекция)")]:
        if img_path and Path(img_path).exists():
            story.append(Paragraph(f"<b>{title}</b>", styles["h3"]))
            # вставка изображения как рисунка фиксированной ширины
            from reportlab.platypus import Image
            story.append(Image(img_path, width=170*mm, height=90*mm))
            story.append(Spacer(1, 6))

    if extra_notes:
        story.append(Paragraph("<b>Примечания</b>", styles["h3"]))
        story.append(Paragraph(extra_notes.replace("\n", "<br/>"), styles["BodyText"]))

    doc = SimpleDocTemplate(filename, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm)
    doc.build(story)
