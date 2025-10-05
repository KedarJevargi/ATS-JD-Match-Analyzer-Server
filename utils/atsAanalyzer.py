import fitz  # PyMuPDF
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
import re
from scipy.signal import find_peaks
import traceback
from io import BytesIO

def extract_skills_from_json(data):
    """
    Recursively traverses a nested dictionary/list structure to extract all string values
    into a single flat set for keyword matching.
    """
    skills = set()
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ["job_title", "experience_years"]: continue
            skills.update(extract_skills_from_json(value))
    elif isinstance(data, list):
        for item in data:
            skills.update(extract_skills_from_json(item))
    elif isinstance(data, str) and data:
        skills.add(data)
    return skills

async def analyze_resume(pdf_path, jd_json_data=None, visualize=True):
    """
    Comprehensive ATS-friendliness checker with robust LaTeX handling,
    advanced keyword matching, and full visualization.
    """
    try:
        # --- Part 1: PDF Parsing ---
        # Handle both file path and BytesIO - CORRECTED
        if isinstance(pdf_path, str):
            doc = fitz.open(pdf_path)
        elif isinstance(pdf_path, BytesIO):
            # Use stream parameter with getvalue() for BytesIO
            doc = fitz.open(stream=pdf_path.getvalue(), filetype="pdf")
        else:
            # Handle other file-like objects
            if hasattr(pdf_path, 'read'):
                content = pdf_path.read()
                doc = fitz.open(stream=content, filetype="pdf")
            else:
                raise ValueError(f"Unsupported input type: {type(pdf_path)}")
        
        line_data, all_fonts, text_alignment_data = [], [], []
        has_images, page_width, full_resume_text = False, 0, ""

        for page in doc:
            page_width = page.rect.width
            full_resume_text += page.get_text()
            if len(page.get_images()) > 0: has_images = True
            
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        spans = line.get("spans", [])
                        line_text = "".join(span.get("text", "") for span in spans).strip()
                        if len(line_text) < 3: continue
                        
                        x_start = min([s["bbox"][0] for s in spans if "bbox" in s])
                        line_data.append((x_start, len(line_text)))
                        
                        for span in spans:
                            all_fonts.append(span.get("font", "N/A"))
                            text_alignment_data.append(span["bbox"][0] / page_width)
        doc.close()

        if len(line_data) < 5:
            return {"error": "Not enough readable text found to analyze."}

        # --- Part 2: Structural Analysis ---
        x_positions = np.array([x for x, _ in line_data])
        total_lines = len(x_positions)
        
        column_groups = 1
        if x_positions.size > 0:
            hist, _ = np.histogram(x_positions, bins=50, range=(0, page_width))
            prominence_threshold = max(np.max(hist) * 0.05, 1) if hist.size > 0 and np.max(hist) > 0 else 1
            candidate_peaks, _ = find_peaks(hist, prominence=prominence_threshold)
            weight_threshold = max(total_lines * 0.05, 5)
            significant_peaks = [p for p in candidate_peaks if hist[p] > weight_threshold]
            column_groups = len(significant_peaks) if len(significant_peaks) > 0 else 1
        
        is_single_column = bool(column_groups <= 1)

        ATS_FRIENDLY_FONTS = {'arial', 'calibri', 'times', 'helvetica', 'georgia', 'garamond', 'cambria', 'verdana', 'tahoma', 'computer modern', 'cmr', 'lmroman'}
        font_compatibility_score = 100.0
        if all_fonts:
            font_counter = Counter(all_fonts)
            ats_friendly_font_count = sum(count for font, count in font_counter.items() if any(ats in font.lower() for ats in ATS_FRIENDLY_FONTS))
            font_compatibility_score = (ats_friendly_font_count / len(all_fonts)) * 100
        uses_simple_fonts = bool(font_compatibility_score > 80)

        no_images = bool(not has_images)
        has_clear_headers = True 
        left_alignment_score = 0.0
        if text_alignment_data:
            left_aligned_count = sum(1 for r in text_alignment_data if r < 0.2)
            left_alignment_score = (left_aligned_count / len(text_alignment_data)) * 100
        is_left_aligned = bool(left_alignment_score > 70)
        no_tables = True

        # --- Part 3: Scoring & Keyword Analysis ---
        structural_checks = [is_single_column, uses_simple_fonts, no_images, has_clear_headers, is_left_aligned, no_tables]
        structural_score = (sum(structural_checks) / len(structural_checks)) * 100

        keyword_match_score, found_skills, missing_skills = 0.0, [], []
        required_skills = set()
        if jd_json_data:
            required_skills = extract_skills_from_json(jd_json_data)
            if required_skills:
                resume_text_norm = " " + full_resume_text.lower().replace('\n', ' ') + " "
                for skill in sorted(list(required_skills)):
                    pattern = r'(?<!\w)' + re.escape(skill.lower()) + r'(?!\w)'
                    if re.search(pattern, resume_text_norm):
                        found_skills.append(skill)
                    else:
                        missing_skills.append(skill)
                keyword_match_score = (len(found_skills) / len(required_skills)) * 100 if required_skills else 0.0
        
        overall_score = (keyword_match_score * 0.6) + (structural_score * 0.4) if jd_json_data else structural_score
        
        # --- Part 4: Visualization ---
        if visualize:
            fig = plt.figure(figsize=(15, 12))
            gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3)
            fig.suptitle('ATS Resume Analysis Report', fontsize=20, fontweight='bold')

            ax1 = fig.add_subplot(gs[0, :])
            if x_positions.size > 0:
                hist, bin_edges = np.histogram(x_positions, bins=50, range=(0, page_width))
                ax1.hist(x_positions, bins=50, color='skyblue', edgecolor='black', alpha=0.7, range=(0, page_width))
                ax1.set_title(f'Layout Analysis: Detected {column_groups} Significant Column(s)', fontweight='bold')
                ax1.set_xlabel('Text Start Position (pixels)')
                ax1.set_ylabel('Frequency')
            else:
                ax1.text(0.5, 0.5, 'No valid text positions for layout analysis.', ha='center', va='center')
                ax1.set_title('Layout Analysis: N/A', fontweight='bold')
            ax1.grid(alpha=0.3)
            
            ax2 = fig.add_subplot(gs[1, 0])
            if jd_json_data and required_skills:
                labels = 'Keywords Matched', 'Keywords Missing'
                sizes = [len(found_skills), len(missing_skills)]
                colors = ['#4CAF50', '#F44336']
                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={"edgecolor":"black"})
                ax2.set_title('Keyword Match vs. Job Description', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No Job Description Provided', ha='center', va='center')
                ax2.set_title('Keyword Match Analysis', fontweight='bold')
            ax2.axis('equal')

            ax3 = fig.add_subplot(gs[1, 1])
            check_names = ['Single Column', 'Simple Fonts', 'No Images', 'Clear Headers', 'Left Aligned', 'No Tables']
            check_values = [c * 100 for c in structural_checks]
            colors = ['#4CAF50' if v > 50 else '#F44336' for v in check_values]
            bars = ax3.barh(check_names, check_values, color=colors, edgecolor='black')
            ax3.set_title('Structural ATS Compliance', fontweight='bold')
            ax3.set_xlabel('Compliance Score (%)')
            ax3.set_xlim(0, 100)
            for bar in bars:
                width = bar.get_width()
                label = '✓ PASS' if width > 50 else '✗ FAIL'
                ax3.text(width / 2, bar.get_y() + bar.get_height()/2, label, ha='center', va='center', color='white', fontweight='bold')

            ax4 = fig.add_subplot(gs[2, :])
            ax4.axis('off')
            scores = {'Structural Score': structural_score, 'Keyword Score': keyword_match_score, 'Overall Match Score': overall_score}
            y_pos = 0.8
            for name, score in scores.items():
                color = 'green' if score >= 75 else 'orange' if score >= 50 else 'red'
                ax4.text(0.5, y_pos, f'{name}: {score:.1f}%', ha='center', fontsize=16, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', ec=color, lw=2))
                y_pos -= 0.35
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # --- Part 5: Final JSON Output ---
        final_json_output = {
            "column": is_single_column,
            "simple fonts": uses_simple_fonts,
            "no images": no_images,
            "clear section header": has_clear_headers,
            "poor text alignment": not is_left_aligned,
            "no tables": no_tables,
            "key words matched": found_skills,
            "keyword missing": missing_skills,
            "score": {
                "overall score": round(overall_score, 2),
                "structure score": round(structural_score, 2),
                "keyword score": round(keyword_match_score, 2)
            }
        }
        return final_json_output

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        return {"error": str(e)}