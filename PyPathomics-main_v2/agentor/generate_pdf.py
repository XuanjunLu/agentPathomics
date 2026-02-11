import os
import pathlib
import uuid
import fitz
import glob
import pandas as pd
from io import BytesIO
import base64
from PIL import Image
from playwright.sync_api import sync_playwright
from jinja2 import Environment, FileSystemLoader

APP_DIR = pathlib.Path(os.path.dirname(__file__)).parent
env = Environment(loader=FileSystemLoader([os.path.join(APP_DIR, 'templates')]))

list_feats_selection_Dict = {
    'Univariate': "UnivariateFeatureSelection",
     'mutualInfo': "mutual_info_selection",
     'mrmr': "mrmr_selection",
     'ttest': "ttest_selection",
     'ranksums': "ranksums_selection",
     'RFE': "rfe_selection",
     'RandomForest': "random_forest_slectiion",
     'Elastic-Net': "elastic_net_slectiion",
     'XGBoost': "XGBoost_slectiion",
     'Lasso': "lasso_feature_selection"}


def get_base64_img(img_path):
    # 读取图片
    image = Image.open(img_path)

    # 将图片转换为字节流
    buffered = BytesIO()
    if img_path.split('.')[-1].upper() == 'PNG':
        image.save(buffered, format="PNG")
    else:
        try:
            image.save(buffered, format="JPEG")
        except Exception as e:
            image.save(buffered, format="PNG")
            # image.convert("RGB").save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    # 将字节流编码为Base64字符串
    base64_img = base64.b64encode(image_bytes)
    base64_img = base64_img.decode('utf-8')
    return 'data:image/png;base64,' + base64_img


def _first_existing_file(candidates, what: str):
    """
    从候选列表中返回第一个存在的文件路径（字符串）。
    candidates: Iterable[Union[str, Path]]
    """
    for p in candidates:
        if p is None:
            continue
        pp = pathlib.Path(p)
        if pp.exists() and pp.is_file():
            return str(pp)
    raise FileNotFoundError(f"未找到 {what}，候选项：{[str(x) for x in candidates if x is not None]}")


def _first_existing_dir(candidates, what: str):
    for p in candidates:
        if p is None:
            continue
        pp = pathlib.Path(p)
        if pp.exists() and pp.is_dir():
            return str(pp)
    raise FileNotFoundError(f"未找到 {what}，候选项：{[str(x) for x in candidates if x is not None]}")


def generate_pdf_func(html_content_str, output='output.pdf'):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(html_content_str)
        # 打印为PDF
        pdf_bytes = page.pdf(format='A4', print_background=True, scale=1,
                 display_header_footer=False, prefer_css_page_size=True,
                 margin={'top': '0', 'bottom': '0', 'left': '0', 'right': '0'})

        # pdf_bytes = page.pdf(format='A4', print_background=True)

        browser.close()
        return pdf_bytes


def pdf_2_img(png_dict, pdfPath, imagePath):
    # png_dict['gene_rank'] =
    res = []
    for item in os.listdir(pdfPath):

        # doc = fitz.open(pdfPath)
        doc = fitz.open(os.path.join(pdfPath, item))
        os.makedirs(imagePath, exist_ok=True)
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap()
            # output_file = os.path.join(imagePath, f"gene_rank.png")
            output_file = os.path.join(imagePath, f"{item.replace('pdf', 'png')}")
            pix.save(output_file)
        res.append(get_base64_img(os.path.join(imagePath, f"{item.replace('pdf', 'png')}")))
    png_dict['gene_rank'] = res[:4]

# package pdf data

data_source_path = r'D:\work_space\AI\PyPathomics-mainv4\PyPathomics-main\example_folder\patient_1757388510134'


def generate_pdf_for_pathomics(data_source_path, llm_res=None, out_put=None, cross_validation_dir=None):
    # Allow explicit cross_validation directory path (fallback to default relative path)
    if cross_validation_dir:
        cv_path = cross_validation_dir
    else:
        cv_path = os.path.join(data_source_path, 'cross_validation')
    
    auc_table = pd.read_table(os.path.join(cv_path, 'auc_table.csv'))
    table_columns = auc_table.columns.tolist()[0].split(",")
    table_rows = []
    t = auc_table.values.tolist()

    for item in t:
        tt = item[0].split(",")
        new_list = [i.replace(' ', '') for i in tt]
        table_rows.append(new_list)

    max_row = auc_table.max().max().split(',')
    best_feats_selection = max_row[0].replace(" ", "")
    index = 0
    max_val = 0
    max_index = 0
    for item in max_row:
        if "/" in item:
            if float(item.split("/")[0].replace(" ", "")) > max_val:
                max_index = index
                max_val = float(item.split("/")[0].replace(" ", ""))
        index += 1
    best_classifers_args = table_columns[max_index].replace(" ", "")

    png_dict = {}
    png_list = glob.glob(os.path.join(cv_path, f"{list_feats_selection_Dict.get(best_feats_selection)}_{best_classifers_args}*.png"))
    for png_file in png_list:
        if "PM" in png_file or "AM" in png_file:
            if "PM_violinplots" in png_file or "AM_violinplots" in png_file:
                png_dict['violinplots'] = get_base64_img(png_file)
            elif "PM_ROC_Curve" in png_file or "AM_ROC_Curve" in png_file:
                png_dict['ROC'] = get_base64_img(png_file)
            else:
                png_dict['top_feature'] = get_base64_img(png_file)
        else:
            png_dict['validate'] = get_base64_img(png_file)

    feature_flag = os.listdir(os.path.join(data_source_path, 'FEA_GSEA'))[0]

    # 雷达图：不同版本命名不一致（可能是每组合一张，也可能只有总图 Radar_charts.png）
    cv_dir = pathlib.Path(cv_path)
    radar_candidates = [
        cv_dir / f"{best_feats_selection}_{best_classifers_args}_Radar_charts.png",
        cv_dir / "Radar_charts.png",
    ] + [pathlib.Path(p) for p in glob.glob(str(cv_dir / "*Radar_charts*.png"))]
    radar_path = _first_existing_file(radar_candidates, what="雷达图(radar)")
    png_dict['radar'] = get_base64_img(radar_path)

    # ROC：旧版本可能保存为 *ROC_Curve.png；当前版本常为 cross_validation/ROC_curve.png
    if not png_dict.get("ROC"):
        roc_candidates = [
            cv_dir / "ROC_curve.png",
        ] + [pathlib.Path(p) for p in glob.glob(str(cv_dir / "*ROC*.png"))]
        try:
            roc_path = _first_existing_file(roc_candidates, what="ROC 曲线(ROC)")
            png_dict["ROC"] = get_base64_img(roc_path)
        except Exception:
            # 保持缺省，由模板显示空图（但不应再抛异常）
            png_dict["ROC"] = ""

    # Kaplan-Meier 曲线：模板使用 png_dict.validate
    # 优先从 cross_validation/km_curve_every 里选“最佳组合”的 KM 图；找不到就退化选任意一张
    if not png_dict.get("validate"):
        km_dir = cv_dir / "km_curve_every"
        km_candidates = []
        if km_dir.exists():
            sel_nm = list_feats_selection_Dict.get(best_feats_selection, best_feats_selection)
            km_candidates += [pathlib.Path(p) for p in glob.glob(str(km_dir / f"{sel_nm}*{best_classifers_args}*.png"))]
            km_candidates += [pathlib.Path(p) for p in glob.glob(str(km_dir / "*.png"))]
        try:
            km_path = _first_existing_file(km_candidates, what="KM 曲线(validate)")
            png_dict["validate"] = get_base64_img(km_path)
        except Exception:
            png_dict["validate"] = ""

    feature_dir = pathlib.Path(data_source_path) / "FEA_GSEA" / feature_flag
    png_dict['BarPlot'] = get_base64_img(str(feature_dir / "BarPlot.png"))
    png_dict['DAG_multiple_terms'] = get_base64_img(str(feature_dir / "DAG_multiple_terms.png"))
    png_dict['HeatPlot'] = get_base64_img(str(feature_dir / "HeatPlot.png"))
    png_dict['RidgePlot'] = get_base64_img(str(feature_dir / "RidgePlot.png"))
    png_dict['TermGeneNet'] = get_base64_img(str(feature_dir / "TermGeneNet.png"))
    png_dict['upsetplot'] = get_base64_img(str(feature_dir / "upsetplot.png"))
    # pdf_2_img(data_source_path + "/FEA_GSEA/GLCM_Correlation_tissue_tissue/GSEA_PyPathomics_fpkm/prerank/Oxidative Phosphorylation.pdf", data_source_path + "/FEA_GSEA/GLCM_Correlation_tissue_tissue")
    pdf_2_img(
        png_dict,
        _first_existing_dir(
            candidates=[
                feature_dir / "GSEA_PyPathomics_fpkm" / "prerank",
                feature_dir / "GSEA(PyPathomics_fpkm)" / "prerank",
            ] + [pathlib.Path(p) for p in glob.glob(str(feature_dir / "**" / "prerank"), recursive=True)],
            what="GSEA prerank 目录",
        ),
        str(feature_dir))
    # png_dict['gene_rank'] = get_base64_img(data_source_path + "/FEA_GSEA/GLCM_Correlation_tissue_tissue/gene_rank.png")
    png_dict['stage_diff'] = get_base64_img(str(feature_dir / "T_stage_difference.png"))

    # with open(r'D:\work_space\AI\PyPathomics-mainv4\PyPathomics-main\example_folder\fc3e0f1b-7f02-11f0-a955-047c16784efc\FEA_GSEA\report_1755831723425.txt', 'r') as f:
    #     llm_res = f.read()
    html_content_str = f"<script>{env.get_template('marked.min.js').render()}</script>"
    html_content_str += env.get_template("page1.html").render(table_columns=table_columns, table_rows=table_rows, png_dict=png_dict)
    html_content_str += env.get_template("page2.html").render(png_dict=png_dict)
    html_content_str += env.get_template("page3.html").render(png_dict=png_dict)
    html_content_str += env.get_template("page4.html").render(llm_res=llm_res)

    pdf_byte = generate_pdf_func(html_content_str)
    if not out_put:
        out_put = f'{uuid.uuid1()}_test.pdf'
    with open(out_put, 'wb') as f:
        f.write(pdf_byte)



if __name__ == '__main__':
    with open("1.txt", 'r', encoding='utf-8') as f:
        llm_res = f.read()
    generate_pdf_for_pathomics(data_source_path, llm_res=llm_res)