"""
Demo Embedding for file_classified.json (Nguyen Minh Hieu CV)
Demonstrates how to use the embedding API with different sections of the CV
"""

import json

from gradio_client import Client


def load_cv_data():
    """Load CV data from file_classified.json"""
    with open("classified_text/file_classified.json", "r", encoding="utf-8") as f:
        return json.load(f)


def demo_batch_embeddings():
    """Demo for /get_batch_embeddings API - returns detailed embeddings with metadata"""
    print("=== DEMO: Batch Embeddings với chi tiết ===")

    # Initialize client
    client = Client("hieuailearning/BAAI_bge_m3_api")

    # Load CV data
    cv_data = load_cv_data()

    # Prepare text chunks for different CV sections
    text_chunks = []

    # Personal information
    contact_info = f"Name: {cv_data['name']}\nEmail: {cv_data['email']}\nPhone: {cv_data['phone']}\nLocation: {cv_data['location']}"
    text_chunks.append(contact_info)

    # Education section
    if cv_data["education"]:
        text_chunks.append(f"Education: {cv_data['education']}")

    # Experience section
    if cv_data["experience"]:
        text_chunks.append(f"Experience: {cv_data['experience']}")

    # Skills section
    if cv_data["skills"]:
        text_chunks.append(f"Skills: {cv_data['skills']}")

    # Projects section
    if cv_data["projects"]:
        text_chunks.append(f"Projects: {cv_data['projects']}")

    # Summary section
    if cv_data["summary"]:
        text_chunks.append(f"Summary: {cv_data['summary']}")

    # Create input text (each chunk on separate line)
    input_text = "\n".join(text_chunks)

    print(f"Đang tạo embeddings cho {len(text_chunks)} phần của CV...")
    print(
        "Các phần bao gồm: Thông tin cá nhân, Học vấn, Kinh nghiệm, Kỹ năng, Dự án, Tóm tắt"
    )

    try:
        result = client.predict(text_input=input_text, api_name="/get_batch_embeddings")
        print("✅ Thành công! Kết quả embedding chi tiết:")
        print(f"Loại kết quả: {type(result)}")
        print(f"Nội dung: {str(result)[:500]}...")  # Show first 500 chars

        return result
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None


def demo_embeddings_only():
    """Demo for /get_embeddings_only API - returns only the embeddings list"""
    print("\n=== DEMO: Embeddings Only (chỉ vector) ===")

    # Initialize client
    client = Client("hieuailearning/BAAI_bge_m3_api")

    # Load CV data
    cv_data = load_cv_data()

    # Create focused text chunks for key sections
    key_sections = []

    # Technical skills
    if cv_data["skills"]:
        key_sections.append(cv_data["skills"])

    # Professional summary
    if cv_data["summary"]:
        key_sections.append(cv_data["summary"])

    # Recent experience
    if cv_data["experience"]:
        key_sections.append(cv_data["experience"])

    # Combine key sections
    focused_text = "\n".join(key_sections)

    print(f"Tạo embeddings cho các phần chính của CV...")

    try:
        result = client.predict(
            text_input=focused_text, api_name="/get_embeddings_only"
        )
        print("✅ Thành công! Danh sách embeddings:")
        print(f"Loại kết quả: {type(result)}")

        # If result is a list of embeddings, show some stats
        if isinstance(result, (list, tuple)):
            print(f"Số lượng embeddings: {len(result)}")
            if result and isinstance(result[0], (list, tuple)):
                print(f"Kích thước mỗi embedding: {len(result[0])}")

        print(f"Nội dung: {str(result)[:300]}...")  # Show first 300 chars

        return result
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None


def demo_skill_specific_embeddings():
    """Demo tạo embeddings cho từng nhóm kỹ năng riêng biệt"""
    print("\n=== DEMO: Embeddings theo từng nhóm kỹ năng ===")

    client = Client("hieuailearning/BAAI_bge_m3_api")
    cv_data = load_cv_data()

    if not cv_data["skills"]:
        print("Không có thông tin kỹ năng")
        return

    # Parse skills into categories
    skills_text = cv_data["skills"]
    skill_categories = []

    # Split by categories (assuming format like "Category:Skills")
    lines = skills_text.split("\n")
    for line in lines:
        if ":" in line and line.strip():
            skill_categories.append(line.strip())

    # Create embeddings for each skill category
    if skill_categories:
        category_text = "\n".join(skill_categories)
        print(f"Tạo embeddings cho {len(skill_categories)} nhóm kỹ năng...")

        try:
            result = client.predict(
                text_input=category_text, api_name="/get_embeddings_only"
            )
            print("✅ Embeddings cho từng nhóm kỹ năng tạo thành công!")
            print(f"Kết quả: {str(result)[:200]}...")

            return result
        except Exception as e:
            print(f"❌ Lỗi: {e}")
    else:
        print("Không thể phân tích nhóm kỹ năng")


def main():
    """Run all embedding demos for Nguyen Minh Hieu's CV"""
    print("🚀 DEMO EMBEDDING API - CV của Nguyễn Minh Hiếu")
    print("=" * 60)

    # Demo 1: Batch embeddings with details
    demo_batch_embeddings()

    # Demo 2: Embeddings only
    demo_embeddings_only()

    # Demo 3: Skill-specific embeddings
    demo_skill_specific_embeddings()

    print("\n✨ Hoàn thành tất cả demo embeddings!")
    print("Tài liệu này cho thấy cách sử dụng API embeddings với dữ liệu CV thực tế.")


if __name__ == "__main__":
    main()
