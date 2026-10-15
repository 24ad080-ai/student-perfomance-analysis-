import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

BASE_DIR = r"C:\Users\kspra\Desktop\student perfomance analysis and prediction"
FIG_DIR = os.path.join(BASE_DIR, "figures")
PDF_PATH = os.path.join(BASE_DIR, "report.pdf")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return  # Suppress headers and footers on title cover page
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Running Header
        self.drawString(54, 11 * inch - 36, "U21ADP05 — Exploratory Data Analysis & Visualization | Assignment II Report")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)
        
        # Running Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, page_text)
        self.drawString(54, 36, "Confidential & Academic Submission — B.Tech Artificial Intelligence & Data Science")
        self.line(54, 46, 8.5 * inch - 54, 46)
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=colors.HexColor('#1B365D'),
        alignment=1, # Center
        spaceAfter=15
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4A607A'),
        alignment=1,
        spaceAfter=30
    )

    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#1B365D'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2C3E50'),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#222222'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=4
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#555555'),
        alignment=1, # Center
        spaceBefore=4,
        spaceAfter=10
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1A252C'),
        backColor=colors.HexColor('#F4F7F9'),
        borderColor=colors.HexColor('#BDC3C7'),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=8,
        spaceAfter=8
    )

    story = []

    # ==================== TITLE PAGE ====================
    story.append(Spacer(1, 40))
    story.append(Paragraph("STUDENT PERFORMANCE ANALYSIS AND PREDICTION USING EXPLORATORY DATA ANALYSIS AND DEEP LEARNING", title_style))
    story.append(Paragraph("A Rigorous Empirical Investigation of Academic Momentum, Socio-Behavioral Correlates, and Multilayer Perceptron Regression on the UCI Mathematics Dataset", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1B365D'), spaceBefore=10, spaceAfter=25))
    
    meta_data = [
        [Paragraph("<b>Course Code & Title:</b>", body_style), Paragraph("U21ADP05 — Exploratory Data Analysis and Visualization", body_style)],
        [Paragraph("<b>Curriculum / Degree:</b>", body_style), Paragraph("B.Tech in Artificial Intelligence and Data Science", body_style)],
        [Paragraph("<b>Assignment Evaluation:</b>", body_style), Paragraph("Assignment II (Comprehensive Capstone - 40 Marks Rubric)", body_style)],
        [Paragraph("<b>Author / Candidate:</b>", body_style), Paragraph("Department of Artificial Intelligence & Data Science", body_style)],
        [Paragraph("<b>Dataset Source:</b>", body_style), Paragraph("UCI Machine Learning Repository (Cortez & Silva, 2008)", body_style)],
        [Paragraph("<b>Target Variable:</b>", body_style), Paragraph("Final Period Grade (G3, Continuous Scale 0–20)", body_style)],
        [Paragraph("<b>Evaluation Date:</b>", body_style), Paragraph("October 2026", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[180, 320])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('PADDING', (0,0), (-1,-1), 7),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(meta_table)

    story.append(Spacer(1, 35))
    abstract_text = (
        "<b>ABSTRACT:</b> Student academic achievement is an essential metric governing educational policy, institutional accreditation, "
        "and proactive pedagogical intervention. This investigation presents an end-to-end data analytics and predictive modeling framework "
        "applied to the benchmark UCI Mathematics Student Performance dataset (N = 395). Employing Ben Fry's seven stages of data visualization, "
        "we analyze demographic, familial, and behavioral predictors of academic success. We formulate final grade estimation (G3) as a continuous "
        "regression problem and engineer a regularized Multilayer Perceptron (MLP) using TensorFlow/Keras. Crucially, we investigate target leakage "
        "by comparing early-stage models (without previous examination scores G1/G2) against late-stage models (with G1/G2). Experimental results "
        "reveal that academic momentum dominates late-stage prediction, where the regularized MLP attains an R² of 0.763 and an MAE of 1.405 points, "
        "substantially outperforming a naive mean baseline (R² = -0.002, MAE = 3.594). In contrast, early-stage behavioral prediction yields an "
        "R² of 0.240 using regularized linear regression, highlighting the complex non-deterministic nature of human scholastic achievement."
    )
    story.append(Paragraph(abstract_text, callout_style))
    story.append(PageBreak())

    # ==================== SECTION 1: INTRODUCTION & PROBLEM STATEMENT ====================
    story.append(Paragraph("1. Introduction, Objectives, and Problem Statement", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))
    
    intro_p1 = (
        "In modern secondary and higher education, academic failure carries profound socioeconomic consequences, including elevated dropout rates, "
        "diminished career prospects, and psychological distress. Conventional educational management systems often diagnose academic insolvency only "
        "after summative examinations conclude, rendering remedial intervention ineffective. Early identification of academically at-risk students "
        "empowers institutional leadership to deploy targeted tutoring, psychological counseling, and parental engagement programs before catastrophic failure occurs."
    )
    story.append(Paragraph(intro_p1, body_style))
    
    story.append(Paragraph("<b>Problem Statement:</b> Given multi-source student records encompassing demographic attributes, familial background, "
                           "lifestyle habits, and longitudinal test marks, can an automated data-driven pipeline reliably analyze behavioral patterns and "
                           "accurately predict a student's final numeric examination grade (G3)?", body_style))

    story.append(Paragraph("<b>Key Project Objectives:</b>", h2_style))
    story.append(Paragraph("• <b>Rigorous Data Auditing:</b> Verify missing values, record duplicates, and address statistical outliers without corrupting valid student ground-truth records.", bullet_style))
    story.append(Paragraph("• <b>Comprehensive Visual Discovery:</b> Execute Fry's seven stages of data visualization to produce at least five distinct, publication-grade analytical figures.", bullet_style))
    story.append(Paragraph("• <b>Parametric and Non-Parametric Profiling:</b> Compute exact statistical indicators (mean, median, standard deviation, IQR, skewness, kurtosis).", bullet_style))
    story.append(Paragraph("• <b>Deep Learning Formulation:</b> Architect and train a deep Multilayer Perceptron (MLP) in TensorFlow/Keras for continuous regression.", bullet_style))
    story.append(Paragraph("• <b>Target Leakage Dissection:</b> Quantify the performance divergence between early-stage prediction (excluding G1 and G2) and late-stage prediction (including G1 and G2).", bullet_style))
    story.append(Paragraph("• <b>Benchmarking and Interpretability:</b> Benchmark deep learning against a naive dummy baseline and OLS regression, interpreting feature contributions via permutation importance.", bullet_style))

    # ==================== SECTION 2: DATASET DESCRIPTION ====================
    story.append(Spacer(1, 10))
    story.append(Paragraph("2. Dataset Source and Feature Dictionary", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))
    
    dataset_p = (
        "The project utilizes the authoritative Portuguese Secondary School Student Performance dataset curated by Paulo Cortez and Alice Silva (2008), "
        "hosted on the University of California, Irvine (UCI) Machine Learning Repository. We analyze the Mathematics cohort (student-mat.csv), comprising "
        "exactly 395 student instances and 33 attributes. The target variable is G3, representing the final course evaluation on a continuous 20-point scale."
    )
    story.append(Paragraph(dataset_p, body_style))

    feat_table_data = [
        [Paragraph("<b>Feature</b>", body_style), Paragraph("<b>Type</b>", body_style), Paragraph("<b>Domain / Range</b>", body_style), Paragraph("<b>Educational Relevance</b>", body_style)],
        [Paragraph("school", body_style), Paragraph("Nominal", body_style), Paragraph("GP (Gabriel Pereira), MS (Mousinho)", body_style), Paragraph("Institutional environment differences", body_style)],
        [Paragraph("sex", body_style), Paragraph("Nominal", body_style), Paragraph("F (Female), M (Male)", body_style), Paragraph("Demographic distribution across STEM", body_style)],
        [Paragraph("age", body_style), Paragraph("Numeric", body_style), Paragraph("15 to 22 years", body_style), Paragraph("Grade-retention and cohort progression", body_style)],
        [Paragraph("address", body_style), Paragraph("Nominal", body_style), Paragraph("U (Urban), R (Rural)", body_style), Paragraph("Geographic accessibility & educational resources", body_style)],
        [Paragraph("Medu / Fedu", body_style), Paragraph("Ordinal", body_style), Paragraph("0 (None) to 4 (Higher Education)", body_style), Paragraph("Parental academic capital and home support", body_style)],
        [Paragraph("studytime", body_style), Paragraph("Ordinal", body_style), Paragraph("1 (<2h), 2 (2-5h), 3 (5-10h), 4 (>10h)", body_style), Paragraph("Weekly dedicated self-study effort", body_style)],
        [Paragraph("failures", body_style), Paragraph("Numeric", body_style), Paragraph("0 to 3 past class failures", body_style), Paragraph("Historical academic struggle and retention", body_style)],
        [Paragraph("schoolsup", body_style), Paragraph("Nominal", body_style), Paragraph("yes, no", body_style), Paragraph("Remedial institutional academic tutoring", body_style)],
        [Paragraph("higher", body_style), Paragraph("Nominal", body_style), Paragraph("yes, no", body_style), Paragraph("Aspiration for tertiary university education", body_style)],
        [Paragraph("Walc", body_style), Paragraph("Ordinal", body_style), Paragraph("1 (Very Low) to 5 (Very High)", body_style), Paragraph("Weekend alcohol consumption lifestyle factor", body_style)],
        [Paragraph("absences", body_style), Paragraph("Numeric", body_style), Paragraph("0 to 75 days", body_style), Paragraph("Truancy, chronic illness, and school disengagement", body_style)],
        [Paragraph("G1 / G2", body_style), Paragraph("Numeric", body_style), Paragraph("0 to 20 marks", body_style), Paragraph("First and second period academic grades", body_style)],
        [Paragraph("G3 (Target)", body_style), Paragraph("Numeric", body_style), Paragraph("0 to 20 marks", body_style), Paragraph("Final summative course examination score", body_style)]
    ]
    feat_table = Table(feat_table_data, colWidths=[70, 55, 175, 200])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(feat_table)
    story.append(PageBreak())

    # ==================== SECTION 3: PREPROCESSING & METHODOLOGY ====================
    story.append(Paragraph("3. Data Preprocessing, Outlier Rationale, and Leakage Prevention", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))
    
    prep_p1 = (
        "<b>Rigorous Data Cleaning:</b> Systematic auditing confirms zero missing cells (sum = 0) and zero duplicate records across all 395 rows. "
        "To ensure production-grade statistical discipline, the following protocols were established:"
    )
    story.append(Paragraph(prep_p1, body_style))
    story.append(Paragraph("• <b>Preservation of Valid Outliers:</b> Traditional statistical filtering often discards points exceeding 1.5 * IQR. In educational analytics, however, a student with 54 absences or a final mark of zero is <i>not</i> a sensor malfunction or typographic error; rather, they represent critical ground-truth cases of chronic truancy or course withdrawal. Deleting these records introduces lethal survival bias, rendering predictive early-warning tools useless in practical school environments.", bullet_style))
    story.append(Paragraph("• <b>Prevention of Data Snooping / Leakage:</b> All transformers (OneHotEncoder for categorical features and StandardScaler for numerical features) were fitted strictly on the training subset (70%, N = 276) and subsequently transformed over the validation (15%, N = 59) and test subsets (15%, N = 60). Scaling parameters (mean and standard deviation) were never computed across the aggregate dataset.", bullet_style))
    story.append(Paragraph("• <b>Target Leakage Isolation (Setting A vs. Setting B):</b> We explicitly bifurcate modeling into Setting A (Pre-examination Early Warning, strictly omitting G1 and G2) and Setting B (Late-Semester Cumulative Estimation, incorporating G1 and G2). This allows transparent evaluation of whether a model learns demographic behavioral traits versus memorizing correlated past grades.", bullet_style))

    # ==================== SECTION 4: STATISTICAL & DISTRIBUTION ANALYSIS ====================
    story.append(Spacer(1, 10))
    story.append(Paragraph("4. Statistical and Distribution Analysis", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))
    
    stats_p = (
        "Statistical profiling reveals key distributional properties of student academic variables. The table below presents exact summary metrics calculated from the raw dataset."
    )
    story.append(Paragraph(stats_p, body_style))

    stat_table_data = [
        [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>G3 (Final)</b>", body_style), Paragraph("<b>G1 (Period 1)</b>", body_style), Paragraph("<b>G2 (Period 2)</b>", body_style), Paragraph("<b>Studytime</b>", body_style), Paragraph("<b>Failures</b>", body_style), Paragraph("<b>Absences</b>", body_style)],
        [Paragraph("Count", body_style), Paragraph("395", body_style), Paragraph("395", body_style), Paragraph("395", body_style), Paragraph("395", body_style), Paragraph("395", body_style), Paragraph("395", body_style)],
        [Paragraph("Mean", body_style), Paragraph("10.415", body_style), Paragraph("10.909", body_style), Paragraph("10.714", body_style), Paragraph("2.035", body_style), Paragraph("0.334", body_style), Paragraph("5.709", body_style)],
        [Paragraph("Std Dev", body_style), Paragraph("4.581", body_style), Paragraph("3.319", body_style), Paragraph("3.762", body_style), Paragraph("0.839", body_style), Paragraph("0.744", body_style), Paragraph("8.003", body_style)],
        [Paragraph("Median", body_style), Paragraph("11.000", body_style), Paragraph("11.000", body_style), Paragraph("11.000", body_style), Paragraph("2.000", body_style), Paragraph("0.000", body_style), Paragraph("4.000", body_style)],
        [Paragraph("IQR (Q3-Q1)", body_style), Paragraph("6.000", body_style), Paragraph("5.000", body_style), Paragraph("4.000", body_style), Paragraph("1.000", body_style), Paragraph("0.000", body_style), Paragraph("8.000", body_style)],
        [Paragraph("Skewness", body_style), Paragraph("-0.733", body_style), Paragraph("+0.241", body_style), Paragraph("-0.432", body_style), Paragraph("+0.632", body_style), Paragraph("+2.387", body_style), Paragraph("+3.672", body_style)],
        [Paragraph("Kurtosis", body_style), Paragraph("+0.403", body_style), Paragraph("-0.700", body_style), Paragraph("+0.601", body_style), Paragraph("-0.014", body_style), Paragraph("+5.674", body_style), Paragraph("+26.237", body_style)]
    ]
    stat_summary_table = Table(stat_table_data, colWidths=[80, 70, 70, 70, 70, 70, 70])
    stat_summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,1), (-1,-1), 'CENTER')
    ]))
    story.append(stat_summary_table)
    
    story.append(Spacer(1, 8))
    dist_note = (
        "<b>Important Inferential Finding:</b> Final grade G3 exhibits negative skewness (-0.733) primarily driven by an anomalous zero-grade cluster "
        "(38 students, 9.62%). The median (11.0) surpasses the mean (10.42). In contrast, absences display severe positive skewness (+3.672) and leptokurtic kurtosis (+26.237), "
        "indicating that while 75% of students accumulate 8 or fewer absences, a long right-tail extends up to 75 missed days. "
        "<i>Critical Statistical Caveat:</i> While correlation metrics quantify associative strength, they do not establish causal mechanisms. For instance, "
        "while past failures correlate negatively with grades (r = -0.36), failure itself is a symptom of underlying cognitive, familial, or socioeconomic stressors."
    )
    story.append(Paragraph(dist_note, callout_style))
    story.append(PageBreak())

    # ==================== SECTION 5: EXPLORATORY VISUALIZATIONS ====================
    story.append(Paragraph("5. Exploratory Visualizations and Empirical Interpretations", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))

    def add_figure_block(fig_name, title, desc, width=420, height=220):
        img_path = os.path.join(FIG_DIR, fig_name)
        if os.path.exists(img_path):
            story.append(KeepTogether([
                Paragraph(f"<b>{title}</b>", h2_style),
                Image(img_path, width=width, height=height),
                Paragraph(desc, caption_style),
                Spacer(1, 6)
            ]))

    add_figure_block(
        "fig1_g3_distribution.png",
        "Figure 1: Target Variable (G3) Statistical Distribution & Modality",
        "Figure 1: Dual boxplot and histogram with KDE of final grades (G3). Annotates the prominent zero-grade cluster (38 students, 9.6%), mean (10.42), and median (11.00)."
    )

    add_figure_block(
        "fig2_studytime_vs_grades_boxplot.png",
        "Figure 2: Weekly Study Time Categories vs. Final Grade Dispersion",
        "Figure 2: Boxplot with overlaid student strip points showing grade distributions across weekly study tiers. Mean scores rise from 10.05 (<2h) to 11.41 (5-10h)."
    )
    story.append(PageBreak())

    add_figure_block(
        "fig3_failures_studytime_countplot.png",
        "Figure 3: Prior Academic Failures and Institutional Remedial Support",
        "Figure 3: Grouped count plot of historical class failures segmented by extra educational support (schoolsup). Highlights that tutoring is deployed early at 0-1 failures."
    )

    add_figure_block(
        "fig4_correlation_heatmap.png",
        "Figure 4: Pearson Correlation Matrix Across All 16 Numerical Attributes",
        "Figure 4: Annotated 16x16 correlation heatmap. Reveals dominant correlations for G2 (r = +0.90), G1 (r = +0.80), past failures (r = -0.36), and parental education (r = +0.22)."
    )
    story.append(PageBreak())

    add_figure_block(
        "fig5_absences_vs_g3_scatter.png",
        "Figure 5: Absenteeism vs. Final Academic Grade with Leverage Outliers",
        "Figure 5: Scatter plot with OLS trendline showing absences vs. G3 (r = +0.034, p = 0.50). High-absence outliers (54, 56, 75 days) maintain passing grades (11-13)."
    )

    add_figure_block(
        "fig6_period_grades_trajectory.png",
        "Figure 6: Longitudinal Grade Progression Across Assessment Periods",
        "Figure 6: Scatterplots comparing G1 and G2 against G3, demonstrating strong academic momentum and persistence across terms."
    )
    story.append(PageBreak())

    add_figure_block(
        "fig7_alcohol_family_impact.png",
        "Figure 7: Socio-Behavioral and Parental Educational Influences",
        "Figure 7: Bivariate analysis of weekend alcohol consumption (Walc) and parental education interaction heatmap showing higher marks when parents hold tertiary degrees."
    )

    # ==================== SECTION 6: DEEP LEARNING MODEL ====================
    story.append(Paragraph("6. Deep Learning Architecture and Modeling Methodology", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))
    
    dl_desc = (
        "We construct a Multilayer Perceptron (MLP) for continuous regression using TensorFlow and Keras. Tabular data problems with small sample sizes "
        "(N = 276 training samples) are notoriously prone to overfitting and covariate shift when subjected to standard deep learning practices (such as Batch Normalization). "
        "Consequently, we design a compact, strictly regularized topology:<br/><br/>"
        "• <b>Input Layer:</b> Receives preprocessed feature vectors (41 dimensions for Setting B, 39 for Setting A).<br/>"
        "• <b>Hidden Layer 1:</b> 32 units, ReLU activation function, L2 weight decay (lambda = 0.015).<br/>"
        "• <b>Dropout Layer 1:</b> Rate = 0.20 to break co-dependent feature representations.<br/>"
        "• <b>Hidden Layer 2:</b> 16 units, ReLU activation function, L2 weight decay (lambda = 0.015).<br/>"
        "• <b>Output Layer:</b> 1 linear unit producing continuous scalar predictions.<br/>"
        "• <b>Optimization & Loss:</b> Adam optimizer (lr = 0.003) minimizing Mean Squared Error (MSE), paired with early stopping (patience = 15) to safeguard generalization."
    )
    story.append(Paragraph(dl_desc, body_style))
    story.append(PageBreak())

    # ==================== SECTION 7: MODEL EVALUATION & RESULTS ====================
    story.append(Paragraph("7. Model Evaluation Results and Visual Diagnostics", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph("Quantitative test set performance was computed strictly on the held-out test split (N = 60). Below is the consolidated benchmark.", body_style))

    results_table_data = [
        [Paragraph("<b>Experiment Setting</b>", body_style), Paragraph("<b>Evaluated Model</b>", body_style), Paragraph("<b>MAE</b>", body_style), Paragraph("<b>MSE</b>", body_style), Paragraph("<b>RMSE</b>", body_style), Paragraph("<b>R² Score</b>", body_style)],
        [Paragraph("Baseline", body_style), Paragraph("Dummy Mean Regressor", body_style), Paragraph("3.594", body_style), Paragraph("20.065", body_style), Paragraph("4.479", body_style), Paragraph("-0.002", body_style)],
        [Paragraph("Setting A (No G1/G2)", body_style), Paragraph("Linear Regression (OLS)", body_style), Paragraph("3.047", body_style), Paragraph("15.214", body_style), Paragraph("3.901", body_style), Paragraph("0.240", body_style)],
        [Paragraph("Setting A (No G1/G2)", body_style), Paragraph("Deep Learning MLP", body_style), Paragraph("3.741", body_style), Paragraph("21.733", body_style), Paragraph("4.662", body_style), Paragraph("-0.085", body_style)],
        [Paragraph("Setting B (With G1/G2)", body_style), Paragraph("Linear Regression (OLS)", body_style), Paragraph("1.516", body_style), Paragraph("4.224", body_style), Paragraph("2.055", body_style), Paragraph("0.789", body_style)],
        [Paragraph("Setting B (With G1/G2)", body_style), Paragraph("Deep Learning MLP", body_style), Paragraph("1.405", body_style), Paragraph("4.747", body_style), Paragraph("2.179", body_style), Paragraph("0.763", body_style)]
    ]
    res_table = Table(results_table_data, colWidths=[110, 140, 60, 60, 60, 70])
    res_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (2,1), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(res_table)
    story.append(Spacer(1, 10))

    add_figure_block(
        "fig8_training_validation_loss.png",
        "Figure 8: Training and Validation MSE Loss Curves vs. Epochs",
        "Figure 8: Convergence trajectory of training and validation loss for the MLP model (Setting B). Illustrates stable convergence without overfitting."
    )

    add_figure_block(
        "fig9_actual_vs_predicted.png",
        "Figure 9: Actual vs. Predicted Test Final Grades",
        "Figure 9: Test predictions vs. actual G3 scores compared against the ideal y = x line (R² = 0.763, MAE = 1.41). Minor dispersion occurs at zero marks."
    )
    story.append(PageBreak())

    add_figure_block(
        "fig10_residual_distribution.png",
        "Figure 10: Histogram and Density of Prediction Residuals",
        "Figure 10: Prediction error distribution (Actual - Predicted). Demonstrates near-zero mean and approximately Gaussian spread across test instances."
    )

    add_figure_block(
        "fig11_residuals_vs_predicted.png",
        "Figure 11: Residuals vs. Fitted Values (Homoscedasticity Assessment)",
        "Figure 11: Residuals plotted against predicted G3 values. Confirms stable error variance across the standard grade range (8 to 18)."
    )
    story.append(PageBreak())

    add_figure_block(
        "fig12_permutation_importance.png",
        "Figure 12: Top 10 Permutation Feature Importances on Test Split",
        "Figure 12: Increase in test MSE when individual feature columns are randomly shuffled. Confirms G2 and G1 as the overwhelming primary determinants."
    )

    # ==================== SECTION 8: CONCLUSION & FUTURE SCOPE ====================
    story.append(Paragraph("8. Conclusion, Critical Limitations, and Future Scope", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))
    
    concl_text = (
        "<b>Synthesis of Core Findings:</b><br/>"
        "1. <b>The Preeminence of Academic Momentum:</b> Academic performance is highly autocorrelated across terms. Incorporating G1 and G2 enables the deep MLP "
        "to account for 76.3% of test variance with an average error of only 1.41 points on a 20-point scale.<br/>"
        "2. <b>The Early-Stage Predictive Gap:</b> Excluding historical grades reduces explanatory power substantially (R² = 0.240 for OLS; -0.085 for MLP). "
        "This proves that demographic and lifestyle attributes provide contextual indicators but cannot deterministically dictate numeric grades.<br/>"
        "3. <b>Deep Learning on Small Tabular Data:</b> Without heavy regularization and target normalization, neural networks overfit on small sample sizes (N = 276). "
        "Linear models remain competitive and interpretable benchmarks in low-sample tabular regimes.<br/><br/>"
        "<b>Project Limitations:</b><br/>"
        "• The sample size is limited to 395 students across two regional schools in Portugal, restricting broad geographic generalization.<br/>"
        "• Self-reported survey attributes (e.g. alcohol consumption, free time) are subject to social desirability bias.<br/>"
        "• The zero-grade inflation represents exam dropouts/absences rather than continuous skill measurement.<br/><br/>"
        "<b>Future Enhancements:</b><br/>"
        "• Developing a two-stage hurdle model: Stage 1 classifies exam attendance (G3 = 0 vs G3 > 0), Stage 2 performs regression on sitters.<br/>"
        "• Integrating multi-task learning to simultaneously predict G1, G2, and G3 with shared latent representations."
    )
    story.append(Paragraph(concl_text, body_style))

    # ==================== SECTION 9: REFERENCES ====================
    story.append(Spacer(1, 10))
    story.append(Paragraph("9. Academic References (IEEE Style)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))
    
    refs = [
        "[1] P. Cortez and A. Silva, 'Using data mining to predict secondary school student performance,' in <i>Proc. 5th Annual Future Business Technology Conf. (FUBUTEC 2008)</i>, Porto, Portugal, pp. 5–12, 2008. [Online]. Available: https://archive.ics.uci.edu/dataset/320/student%2Bperformance",
        "[2] I. Goodfellow, Y. Bengio, and A. Courville, <i>Deep Learning</i>. Cambridge, MA, USA: MIT Press, 2016. [Online]. Available: https://www.deeplearningbook.org/",
        "[3] B. Fry, <i>Visualizing Data: Exploring and Explaining Data with the Processing Environment</i>. Sebastopol, CA, USA: O'Reilly Media, 2008.",
        "[4] F. Chollet, <i>Deep Learning with Python</i>, 2nd ed. Shelter Island, NY, USA: Manning Publications, 2021.",
        "[5] C. Romero and S. Ventura, 'Educational data mining: A review of the state of the art,' <i>IEEE Trans. Syst., Man, Cybern. C, Appl. Rev.</i>, vol. 40, no. 6, pp. 601–618, Nov. 2010. DOI: 10.1109/TSMCC.2010.2053532"
    ]
    for r in refs:
        story.append(Paragraph(r, bullet_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {PDF_PATH}")

if __name__ == '__main__':
    build_pdf()
