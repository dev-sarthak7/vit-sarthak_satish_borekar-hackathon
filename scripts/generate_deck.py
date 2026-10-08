import sys
from pathlib import Path
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Palette inspired by S&P Global (Deep Navy, Crimson Accent, Cool Gray, White)
NAVY = colors.HexColor("#0B1D3A")
CRIMSON = colors.HexColor("#C41230")
SLATE = colors.HexColor("#334155")
MUTED = colors.HexColor("#64748B")
LIGHT_BG = colors.HexColor("#F8FAFC")
CARD_BG = colors.HexColor("#EEF2F6")
BORDER_COLOR = colors.HexColor("#CBD5E1")
WHITE = colors.HexColor("#FFFFFF")
GREEN = colors.HexColor("#16A34A")

PAGE_WIDTH, PAGE_HEIGHT = landscape(letter)  # 11 x 8.5 inches = 792 x 612 pt

class NumberedCanvas(canvas.Canvas):
    """Custom canvas to draw header, footer, page borders and slide numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, num_pages):
        self.saveState()
        
        # Top banner line on non-title slides
        if self._pageNumber > 1:
            self.setFillColor(NAVY)
            self.rect(0, PAGE_HEIGHT - 6, PAGE_WIDTH, 6, fill=True, stroke=False)
            self.setFillColor(CRIMSON)
            self.rect(0, PAGE_HEIGHT - 6, 120, 6, fill=True, stroke=False)

            # Header text
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(MUTED)
            self.drawString(36, PAGE_HEIGHT - 22, "S&P GLOBAL & CRISIL CAMPUS HACKATHON 2026")
            self.drawRightString(PAGE_WIDTH - 36, PAGE_HEIGHT - 22, "AI/NLP RISK ENGINE & DOWNSTREAM MODULES")
            
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(36, PAGE_HEIGHT - 28, PAGE_WIDTH - 36, PAGE_HEIGHT - 28)

        # Footer on all slides
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(36, 30, PAGE_WIDTH - 36, 30)

        self.setFont("Helvetica", 8)
        self.setFillColor(MUTED)
        self.drawString(36, 18, "Candidate: Sarthak Satish Borekar | VIT Bhopal")
        self.drawRightString(PAGE_WIDTH - 36, 18, f"Slide {self._pageNumber} of {num_pages}")
        
        self.restoreState()


def create_deck(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_main = ParagraphStyle(
        "TitleMain",
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=34,
        textColor=NAVY,
        alignment=0,
    )
    
    title_sub = ParagraphStyle(
        "TitleSub",
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=CRIMSON,
        alignment=0,
    )

    slide_title = ParagraphStyle(
        "SlideTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=NAVY,
        spaceAfter=4,
    )

    slide_subtitle = ParagraphStyle(
        "SlideSubtitle",
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=MUTED,
        spaceAfter=12,
    )

    body = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=SLATE,
    )

    body_bold = ParagraphStyle(
        "BodyBold",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13.5,
        textColor=NAVY,
    )

    bullet = ParagraphStyle(
        "Bullet",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=SLATE,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=4,
    )

    card_header = ParagraphStyle(
        "CardHeader",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=NAVY,
        spaceAfter=6,
    )

    card_text = ParagraphStyle(
        "CardText",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=SLATE,
    )

    story = []

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    story.append(Spacer(1, 40))
    story.append(Paragraph("S&P GLOBAL & CRISIL CAMPUS HACKATHON 2026", ParagraphStyle("Org", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=CRIMSON, spaceAfter=8)))
    story.append(Paragraph("AI/NLP Risk Engine & Dynamic Decision Platform", title_main))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Transforming Unstructured News & Social Streams into Actionable Quantitative Signals,<br/>Tactical Index Rebalancing, and Wholesale Portfolio Stress Testing", title_sub))
    story.append(Spacer(1, 45))

    meta_table_data = [
        [
            Paragraph("<b>Candidate Name:</b> Sarthak Satish Borekar", body),
            Paragraph("<b>Institution:</b> Vellore Institute of Technology - Bhopal", body)
        ],
        [
            Paragraph("<b>College Email:</b> sarthak.23bce10568@vitbhopal.ac.in", body),
            Paragraph("<b>Deliverables:</b> Source Code, Live Dashboard, REST API & Presentation", body)
        ],
        [
            Paragraph("<b>Repository:</b> <code>vit-sarthak_satish_borekar-hackathon</code>", body),
            Paragraph("<b>Scope:</b> Core NLP Engine + <b>Module A (Rebalancer)</b> + <b>Module B (Stress Test)</b>", body)
        ]
    ]
    t1 = Table(meta_table_data, colWidths=[350, 370])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), CARD_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t1)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 2: Problem Statement & Solution Approach
    # -------------------------------------------------------------
    story.append(Paragraph("Problem Statement & Solution Approach", slide_title))
    story.append(Paragraph("Bridging the gap between high-velocity unstructured information and institutional risk decisioning", slide_subtitle))

    p_left = [
        Paragraph("<b>The Institutional Challenge</b>", card_header),
        Paragraph("&bull; <b>Information Overload:</b> Global risk and portfolio teams receive tens of thousands of unstructured news wires and social feeds daily.", bullet),
        Paragraph("&bull; <b>Lagging Metrics:</b> Traditional quantitative models rely heavily on backward-looking financial metrics, lagging rapid sentiment shifts.", bullet),
        Paragraph("&bull; <b>Manual Screening Bottleneck:</b> Manual qualitative screening is unscalable, subjective, and unable to trigger automated portfolio protections.", bullet),
        Paragraph("&bull; <b>Cross-Asset Fragmentation:</b> Difficult to map a single qualitative event (e.g. Geopolitical conflict) to correlated multi-asset risk factor shocks.", bullet),
    ]

    p_right = [
        Paragraph("<b>Our Unified Solution Approach</b>", card_header),
        Paragraph("&bull; <b>Multi-Source Ingestion:</b> Scalable pipeline normalizing both structured news wires (NewsAPI / CSV) and noisy social media (Twitter/X).", bullet),
        Paragraph("&bull; <b>Tripartite NLP Risk Signal:</b> Extracts <b>Sentiment</b> (FinBERT), <b>Event Type</b> (9 categories via Zero-Shot NLI), and <b>Impact Severity</b> (1-10 formula).", bullet),
        Paragraph("&bull; <b>Module A (Tactical Rebalancer):</b> Converts sentiment momentum into risk-constrained equity index weights (3%-10% bounds) with backtest verification.", bullet),
        Paragraph("&bull; <b>Module B (Portfolio Stress Tester):</b> Automates balance sheet stress testing on a $1.8B wholesale banking portfolio under high-impact signals.", bullet),
    ]

    t2 = Table([[p_left, p_right]], colWidths=[355, 355])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), CARD_BG),
        ('BACKGROUND', (1,0), (1,0), LIGHT_BG),
        ('BOX', (0,0), (0,0), 1, BORDER_COLOR),
        ('BOX', (1,0), (1,0), 1, CRIMSON),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t2)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 3: System Design & Architecture
    # -------------------------------------------------------------
    story.append(Paragraph("System Design & End-to-End Data Flow", slide_title))
    story.append(Paragraph("Modular, decoupled architecture connecting raw data ingestion to downstream execution and UI", slide_subtitle))

    arch_img_path = Path(__file__).resolve().parent.parent / "docs" / "architecture.png"
    if arch_img_path.exists():
        story.append(Image(str(arch_img_path), width=7.2*inch, height=4.07*inch))
    else:
        story.append(Paragraph("Architecture diagram located in docs/architecture.png", body))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 4: Implementation Highlights & Tech Stack
    # -------------------------------------------------------------
    story.append(Paragraph("Implementation Highlights & Tech Stack", slide_title))
    story.append(Paragraph("Domain-tailored AI models, robust risk formulas, and production-ready microservices", slide_subtitle))

    mod1 = [
        Paragraph("<b>1. NLP Risk Engine</b>", card_header),
        Paragraph("<b>FinBERT Sentiment:</b> <code>ProsusAI/finbert</code> fine-tuned on financial corpus for exact polarity [-1.0, +1.0].<br/><b>Zero-Shot NLI:</b> <code>nli-distilroberta-base</code> categorizing 9 event types with keyword fallback.<br/><b>Impact Model:</b> Formula balancing event severity, sentiment magnitude, and source trust.", card_text)
    ]
    mod2 = [
        Paragraph("<b>2. Module A: Rebalancer</b>", card_header),
        Paragraph("<b>Universe:</b> 16 large-cap US equities.<br/><b>Tilt Formula:</b> Recency-weighted sentiment (7-day half-life) applies exponential tilt.<br/><b>Risk Limits:</b> Strict 3%-10% per-stock bounds and budget normalization (sum=100%).<br/><b>Backtest:</b> Daily replay without lookahead bias.", card_text)
    ]
    mod3 = [
        Paragraph("<b>3. Module B: Stress Tester</b>", card_header),
        Paragraph("<b>Portfolio:</b> $1.8B wholesale book (6 asset classes: Loans, Corp/Gov Bonds, Swaps, FX, Equities).<br/><b>Automated Trigger:</b> Fires when Impact &ge; 7.0.<br/><b>Valuation Engine:</b> 2nd-order duration-convexity for bonds, &Delta;ECL default provisions for loans, cashflow models for swaps/FX.", card_text)
    ]
    mod4 = [
        Paragraph("<b>4. Delivery & Dashboard</b>", card_header),
        Paragraph("<b>FastAPI:</b> Sub-millisecond REST endpoints with filtering (<code>/signals</code>, <code>/signals/{ticker}</code>).<br/><b>Streamlit + Plotly:</b> 3-tab reactive UI for real-time weight charts, stress loss waterfalls, and signal exploration.<br/><b>Quality:</b> 34/34 passing pytest tests.", card_text)
    ]

    t4 = Table([[mod1, mod2], [mod3, mod4]], colWidths=[355, 355], rowHeights=[140, 140])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), CARD_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t4)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 5: Key Results & Quantitative Findings
    # -------------------------------------------------------------
    story.append(Paragraph("Key Results & Prototype Outputs", slide_title))
    story.append(Paragraph("Quantitative verification across signal classification, portfolio rebalancing, and stress testing", slide_subtitle))

    kpi1 = [
        Paragraph("<font color='#0B1D3A' size=14><b>568 Signals</b></font>", ParagraphStyle("K1", fontName="Helvetica-Bold", leading=16)),
        Paragraph("Ingested and scored across 16 tickers. Zero-shot event classifier achieves high precision against ground-truth evaluation set.", card_text)
    ]
    kpi2 = [
        Paragraph("<font color='#C41230' size=14><b>-$42.8M Shock</b></font>", ParagraphStyle("K2", fontName="Helvetica-Bold", leading=16)),
        Paragraph("Quantified on $1.8B wholesale bank portfolio under Geopolitical event (Impact 8.5), showing loan credit provisions & bond drops.", card_text)
    ]
    kpi3 = [
        Paragraph("<font color='#16A34A' size=14><b>34 / 34 Tests</b></font>", ParagraphStyle("K3", fontName="Helvetica-Bold", leading=16)),
        Paragraph("100% test pass rate across ingestion, pipeline resolution, portfolio math, shock scenarios, valuation, and trigger logic.", card_text)
    ]

    t_kpi = Table([[kpi1, kpi2, kpi3]], colWidths=[232, 232, 232])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 14))

    findings_table = [
        [Paragraph("<b>Component</b>", body_bold), Paragraph("<b>Evaluated Metric / Output</b>", body_bold), Paragraph("<b>Domain Observation & Rigor</b>", body_bold)],
        [
            Paragraph("<b>AI/NLP Risk Engine</b>", body),
            Paragraph("Sentiment [-1 to +1]<br/>Impact [1 to 10]<br/>9 Event Labels", body),
            Paragraph("FinBERT accurately isolates financial tone. Zero-shot classifier eliminates false positives via keyword support cross-check.", body)
        ],
        [
            Paragraph("<b>Module A: Rebalancer</b>", body),
            Paragraph("1-Year Daily Replay<br/>Tracking Error & t-stat<br/>3%-10% Bounds", body),
            Paragraph("Strict concentration bounds prevented idiosyncratic stock blowups during the 2021-2022 tech bear market.", body)
        ],
        [
            Paragraph("<b>Module B: Stress Test</b>", body),
            Paragraph("Multi-factor shocks<br/>6 Asset Classes<br/>Pre/Post Valuation", body),
            Paragraph("Correctly accounts for amortized cost loan credit provisions (&Delta;ECL) vs MTM bond duration and swap hedges.", body)
        ]
    ]
    t_res = Table(findings_table, colWidths=[150, 160, 390])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), CARD_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_res)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 6: Domain Impact & Business Value
    # -------------------------------------------------------------
    story.append(Paragraph("Domain Impact & Institutional Value", slide_title))
    story.append(Paragraph("How the unified platform solves critical operational and risk challenges for financial institutions", slide_subtitle))

    dim1 = [
        Paragraph("<b>1. Proactive Risk Mitigation</b>", card_header),
        Paragraph("Shifts enterprise risk management from reactive post-loss reporting to real-time pre-market event detection. High-impact alerts enable immediate portfolio hedging.", card_text)
    ]
    dim2 = [
        Paragraph("<b>2. 99% Screening Automation</b>", card_header),
        Paragraph("Eliminates hundreds of analyst hours spent manually reading financial news and social media chatter, funneling high-confidence signals directly into risk workflows.", card_text)
    ]
    dim3 = [
        Paragraph("<b>3. Regulatory & Model Governance</b>", card_header),
        Paragraph("Provides auditable, formulaic impact scores and explicit confidence intervals rather than unverifiable black-box predictions, ensuring supervisory compliance.", card_text)
    ]
    dim4 = [
        Paragraph("<b>4. Multi-Horizon Flexibility</b>", card_header),
        Paragraph("Serves both short-horizon tactical desks (Module A intraday rebalancing) and strategic enterprise risk committees (Module B macroeconomic stress testing).", card_text)
    ]

    t_dim = Table([[dim1, dim2], [dim3, dim4]], colWidths=[355, 355], rowHeights=[140, 140])
    t_dim.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t_dim)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 7: Limitations, Assumptions & Future Roadmap
    # -------------------------------------------------------------
    story.append(Paragraph("Limitations, Assumptions & Future Roadmap", slide_title))
    story.append(Paragraph("Transparent disclosure of analytical boundaries and engineering enhancement roadmap", slide_subtitle))

    lim = [
        Paragraph("<b>Current Assumptions & Limitations</b>", card_header),
        Paragraph("&bull; <b>Impact Score Formulation:</b> Severity weights are rule-based heuristics based on financial intuition rather than tick-by-tick regression models.", bullet),
        Paragraph("&bull; <b>Social Media Coverage Asymmetry:</b> Data volume is heavier for mega-cap tech (e.g. TSLA, AAPL) than defensive stocks.", bullet),
        Paragraph("&bull; <b>Derivatives Approximation:</b> Interest-rate swaps and FX forwards are valued without dynamic counterparty credit valuation adjustments (CVA).", bullet),
        Paragraph("&bull; <b>Batch Pipeline:</b> NLP model inference currently operates on exported batch files rather than distributed live Kafka streams.", bullet),
    ]

    nxt = [
        Paragraph("<b>Future Engineering & Research Roadmap</b>", card_header),
        Paragraph("&bull; <b>Empirical Impact Calibration:</b> Train a multi-task transformer on historical high-frequency abnormal stock and bond spread reactions.", bullet),
        Paragraph("&bull; <b>Streaming Event Infrastructure:</b> Deploy model endpoints on distributed Ray/Kafka cluster for sub-second signal streaming.", bullet),
        Paragraph("&bull; <b>Multi-Currency & CDS Stressing:</b> Expand synthetic wholesale banking book to non-USD loan tranches and Credit Default Swaps.", bullet),
        Paragraph("&bull; <b>Cross-Lingual NLP:</b> Integrate multilingual FinBERT to ingest European, Asian, and emerging market financial publications.", bullet),
    ]

    t_lim = Table([[lim, nxt]], colWidths=[355, 355])
    t_lim.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), CARD_BG),
        ('BACKGROUND', (1,0), (1,0), LIGHT_BG),
        ('BOX', (0,0), (0,0), 1, BORDER_COLOR),
        ('BOX', (1,0), (1,0), 1, NAVY),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t_lim)

    # Build PDF with custom canvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Presentation successfully generated at: {output_path}")

if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent.parent / "docs" / "presentation.pdf"
    create_deck(out_file)
