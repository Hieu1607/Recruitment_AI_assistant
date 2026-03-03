import json
import os

from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()


class CVClassifier:
    def __init__(self):
        """Initialize the CV classifier with Groq API client."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=api_key)

    def classify_cv_sections(self, cv_text):
        """
        Classify sections in a CV text using Groq LLM.

        Args:
            cv_text (str): The extracted text from CV

        Returns:
            dict: JSON object with classified sections
        """
        # Define the prompt for section classification
        prompt = (
            """
        You are an expert CV/Resume parser. Please analyze the following CV text and classify it into sections AND extract specific information.

        SECTION CLASSIFICATION:
        - education: Educational background, degrees, schools, certifications
        - experience: Professional experience, work history, employment
        - skills: Technical skills, programming languages, tools, competencies
        - projects: Personal projects, academic projects, portfolio items
        - summary: Professional summary, objective, introduction
        - contact: Contact information, personal details, location
        - achievements: Awards, honors, competitions, recognitions
        - languages: Language proficiency, certifications about these language
        - publications: Research papers, articles, publications
        - certifications: Professional certifications, licenses (dont put language certifications here, put them in languages section)
        - references: References or reference information
        - other: Any other relevant information that doesn't fit the above categories

        SPECIFIC INFORMATION EXTRACTION:
        - educated: Boolean - Did the person educate or not? (true/false, false if no education section or no mention of education found)
        - cpa: Extract CPA, or academic performance metrics (e.g. "3.8 CPA")
        - major: Extract field of study, specialization, or major (e.g., "Computer Science", "Physics")
        - current_job_title: Extract the most recent or current job title/position
        - name: Extract the person's full name
        - location: Extract current location/address, (standadize to province, nationality, or city if possible, eg Hanoi, Vietnam or Hanoi, or Vietnam)
        - phone: Extract phone number
        - email: Extract email address
        - experiment_years: Extract the number total years of experience (e.g., 5, 1.5 ...)

        For each section found, extract the relevant text content. If a section is not present, set it to null.
        For specific information, extract the exact value or set to null if not found.

        Please return the result as a JSON object with the following structure:
        {
            "name": "full name or null",
            "phone": "extracted phone number or null",
            "email": "extracted email address or null",
            "location": "current location or null",
            "contact": "extracted contact content or null",
            "current_job_title": "most recent job title or null",
            "educated": true/false,
            "major": "field of study or null",
            "cpa": "CPA value or null",
            "education": "extracted education content or null",
            "experience": "extracted experience content or null", 
            "experiment_years": "total years of experience or null"
            "skills": "extracted skills content or null",
            "languages": "extracted languages content or null",
            "projects": "extracted projects content or null",
            "summary": "extracted summary content or null",
            "achievements": "extracted achievements content or null",
            "publications": "extracted publications content or null",
            "certifications": "extracted certifications content or null",
            "references": "extracted references content or null",
            "other": "any other relevant content or null"
        }

        Here is the CV text to analyze:

        """
            + cv_text
        )

        try:
            # Send request to Groq API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",  # You can change this to other available models
                temperature=0.1,  # Low temperature for more consistent results
                max_tokens=2048,
            )

            # Extract the response content
            response_content = chat_completion.choices[0].message.content

            # Try to extract JSON from the response
            try:
                # Find JSON in the response (sometimes the model wraps it in markdown)
                start_idx = response_content.find("{")
                end_idx = response_content.rfind("}") + 1

                if start_idx != -1 and end_idx != 0:
                    json_str = response_content[start_idx:end_idx]
                    classified_sections = json.loads(json_str)
                else:
                    # If no JSON found, return the raw response
                    classified_sections = {"raw_response": response_content}

            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw response
                classified_sections = {"raw_response": response_content}

            return classified_sections

        except Exception as e:
            return {"error": f"Failed to classify CV sections: {str(e)}"}

    def extract_specific_info(self, cv_text):
        """
        Extract specific information from CV text (name, education status, GPA, major, job title, location).

        Args:
            cv_text (str): The extracted text from CV

        Returns:
            dict: Specific extracted information
        """
        prompt = f"""
        You are an expert CV/Resume parser. Please extract the following specific information from the CV text:

        1. Name: The person's full name
        2. Educated: Does the person have formal education? (true/false)
        3. GPA/CPA: Any academic performance metrics (GPA, CPA, etc.)
        4. Major: Field of study, specialization, or degree type
        5. Current Job Title: Most recent or current position/job title
        6. Location: Current location or address

        Return the result as a JSON object:
        {{
            "name": "full name or null",
            "educated": true/false,
            "gpa_cpa": "GPA/CPA value or null",
            "major": "field of study or null", 
            "current_job_title": "most recent job title or null",
            "location": "current location or null"
        }}

        CV Text:
        {cv_text}
        """

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=1024,
            )

            response_content = chat_completion.choices[0].message.content

            try:
                start_idx = response_content.find("{")
                end_idx = response_content.rfind("}") + 1

                if start_idx != -1 and end_idx != 0:
                    json_str = response_content[start_idx:end_idx]
                    specific_info = json.loads(json_str)
                else:
                    specific_info = {"raw_response": response_content}

            except json.JSONDecodeError:
                specific_info = {"raw_response": response_content}

            return specific_info

        except Exception as e:
            return {"error": f"Failed to extract specific info: {str(e)}"}

    def process_cv_file(self, input_file_path, output_file_path=None):
        """
        Process a CV text file and save classified sections to JSON.

        Args:
            input_file_path (str): Path to the input text file
            output_file_path (str): Path for the output JSON file (optional)

        Returns:
            dict: Classified sections
        """
        try:
            # Read the CV text file
            with open(input_file_path, "r", encoding="utf-8") as file:
                cv_text = file.read()

            print(f"Processing CV file: {input_file_path}")
            print(f"CV text length: {len(cv_text)} characters")

            # Classify sections
            classified_sections = self.classify_cv_sections(cv_text)

            # Determine output file path
            if output_file_path is None:
                base_name = os.path.splitext(input_file_path)[0]
                output_file_path = f"{base_name}_classified.json"

            # Save results to JSON file
            with open(output_file_path, "w", encoding="utf-8") as file:
                json.dump(classified_sections, file, indent=2, ensure_ascii=False)

            print(f"Classification results saved to: {output_file_path}")
            return classified_sections

        except FileNotFoundError:
            error_msg = f"Input file not found: {input_file_path}"
            print(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error processing CV file: {str(e)}"
            print(error_msg)
            return {"error": error_msg}


def main():
    """Main function to demonstrate the CV classifier."""
    classifier = CVClassifier()

    # Example usage - you can modify these file paths
    input_files = ["Khang.txt", "file.txt"]

    for input_file in input_files:
        if os.path.exists(input_file):
            print(f"\n{'='*50}")
            print(f"Processing: {input_file}")
            print("=" * 50)

            # Full classification with sections and specific info
            result = classifier.process_cv_file(input_file)

            # Print a summary of classified sections
            if "error" not in result:
                print("\nClassified sections found:")
                if "sections" in result:
                    for section, content in result["sections"].items():
                        if content and content != "null":
                            print(
                                f"  ✓ {section.title()}: {len(str(content))} characters"
                            )
                        else:
                            print(f"  ✗ {section.title()}: Not found")

                print("\nExtracted specific information:")
                if "extracted_info" in result:
                    for key, value in result["extracted_info"].items():
                        if value and value != "null":
                            print(f"  ✓ {key.replace('_', ' ').title()}: {value}")
                        else:
                            print(f"  ✗ {key.replace('_', ' ').title()}: Not found")

                # Backward compatibility - if old format is returned
                if "sections" not in result and "extracted_info" not in result:
                    for section, content in result.items():
                        if content and content != "null":
                            print(
                                f"  ✓ {section.title()}: {len(str(content))} characters"
                            )
                        else:
                            print(f"  ✗ {section.title()}: Not found")
            else:
                print(f"Error: {result['error']}")

            # Also demonstrate specific info extraction only
            print(f"\n{'-'*30}")
            print("Quick Info Extraction:")
            print("-" * 30)

            try:
                with open(input_file, "r", encoding="utf-8") as file:
                    cv_text = file.read()

                quick_info = classifier.extract_specific_info(cv_text)
                if "error" not in quick_info:
                    for key, value in quick_info.items():
                        if key != "raw_response":
                            print(
                                f"{key.replace('_', ' ').title()}: {value if value else 'Not found'}"
                            )
                else:
                    print(f"Error in quick extraction: {quick_info['error']}")
            except Exception as e:
                print(f"Error reading file for quick extraction: {e}")
        else:
            print(f"File not found: {input_file}")


if __name__ == "__main__":
    main()
