# https://youtu.be/0qXQe3CDTz4
"""
Google Scholar Literature Search Script
------------------------------------

This script performs automated literature searches on Google Scholar for specified keywords,
extracting detailed information including titles, authors, citations, and intelligent extraction of
methods and fields.

Search Capabilities:
- Supports logical operators:
- Date range filtering
- Multiple pages per search
- Random delays to prevent blocking

Method Extraction Logic:
The script intelligently identifies method-related content from abstracts using:
1. Method-specific keywords: ['using', 'method', 'approach', 'technique', 'protocol', 
                            'workflow', 'analysis', 'procedure', 'methodology']
2. Sentence-level analysis: Extracts complete sentences containing method keywords
3. Context-aware extraction: Maintains complete methodological descriptions

Field Classification Rules:
Automatically categorizes papers into fields based on keyword presence:
1. Biology: ['biology', 'biological', 'cell', 'molecular', 'genetics', 'neuron', 'brain', 'life sciences']
2. Medicine: ['medical', 'clinical', 'healthcare', 'patient', 'treatment', 'cancer', 'disease']
3. Materials: ['microstructure', 'nanostructure', 'alloy', 'grain size']
4. Microscopy: ['microscopy', 'imaging', 'microscope', 'visualization']
5. AI: ['machine learning', 'deep learning', 'computer vision']

Data Extraction:
- Title: Cleaned and formatted
- Authors: Extracted from citation information
- Year: Regex-matched from publication info
- Journal: Parsed from source information
- DOI: Regex-matched from URLs and text
- Methods: Intelligently extracted from abstract
- Field: Determined by keyword analysis
- URLs: Direct links to papers
- Citations: Count of paper citations


Example keyword search terms:
   
Keywords = [
    '"digitalsreeni" AND "youtube"',    # 
    '"Sreenivas Bhattiprolu"',                       # Exact name search
    '"Sreenivas Bhattiprolu" AND "ZEISS"'       
]


Note: Includes anti-blocking measures and error handling to manage 
Google Scholar's rate limiting and robot detection.
"""


import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import urllib3
import re
import random

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def extract_doi(text):
    """
    Extract DOI from text or URL:
        DOI pattern starts with 10, followed by 4 to 9 digits, a / (slash), and allowed characters like -, ., _, ;, (), :
    
    """
    doi_pattern = r'10\.\d{4,9}/[-._;()/:\w]+'   # define a regular expression to match the structure of a DOI.
    match = re.search(doi_pattern, text)   # search the input text for the first match of the specified pattern.
    return match.group(0) if match else ""

def extract_methods(abstract):
    """
    Extract potential methods from abstract
    Extract sentences from an abstract containing specific method-related keywords. 
    Return these sentences joined by semicolons as a single string.
    
    """
    method_keywords = ['using', 'method', 'approach', 'technique', 'protocol', 
                      'workflow', 'analysis', 'procedure', 'methodology']
    
    sentences = abstract.split('.')
    method_sentences = []
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in method_keywords):
            method_sentences.append(sentence.strip())
    
    return '; '.join(method_sentences)

def extract_field(author_info, title, abstract):
    """Extract potential field of study"""
    fields = {
        'biology': ['biology', 'biological', 'cell', 'molecular', 'genetics', 'neuron', 'brain', 'life sciences'],
        'medicine': ['medical', 'clinical', 'healthcare', 'patient', 'treatment', 'cancer', 'disease'],
        'materials': ['microstructure', 'nanostructure', 'alloy', 'grain size'],
        'microscopy': ['microscopy', 'imaging', 'microscope', 'visualization'],
        'AI': ['machine learning', 'deep learning', 'computer vision']
    }
    
    
    
    text = f"{title} {abstract} {author_info}".lower()
    
    detected_fields = []
    for field, keywords in fields.items():
        if any(keyword in text for keyword in keywords):
            detected_fields.append(field)
    
    return '; '.join(detected_fields) if detected_fields else "Not specified"

def format_search_query(keyword, start_year, end_year):
    """
    Format search query with proper logical operators for Google Scholar
    """
    # Remove quotes if they exist and split by OR
    terms = [term.strip(' "\'') for term in keyword.split(' OR ')]
    
    # Format each term
    formatted_terms = []
    for term in terms:
        # If term contains spaces, wrap it in quotes
        if ' ' in term:
            formatted_terms.append(f'"{term}"')
        else:
            formatted_terms.append(term)
    
    # Join terms with OR operator
    query = ' OR '.join(formatted_terms)
    
    # Add date range
    query += f' after:{start_year-1} before:{end_year+1}'
    
    print(f"Formatted query: {query}")  # Debug print
    return query


def search_scholar(keyword, start_year=2020, end_year=2024, num_pages=3):
    """
    Google Scholar search function 
    """
    results = []
    
    for page in range(num_pages):
        start_index = page * 10
        query = format_search_query(keyword, start_year, end_year)
        
        # URL encode the query properly
        encoded_query = requests.utils.quote(query)
        url = f"https://scholar.google.com/scholar?start={start_index}&q={encoded_query}&hl=en"
        
        print(f"\nSearching for: {keyword} - Page {page + 1}")
        print(f"URL: {url}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        try:
            # Add random delay between pages
            if page > 0:
                delay = random.uniform(20, 40)
                print(f"Waiting {delay:.1f} seconds before next page...")
                time.sleep(delay)
                
            response = requests.get(url, headers=headers, verify=False)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Debug print
                print(f"Response content preview: {response.text[:500]}")
                
                if "Please show you're not a robot" in response.text:
                    print("CAPTCHA detected. Please try again later.")
                    break
                
                articles = soup.find_all("div", class_="gs_r gs_or gs_scl")
                print(f"Found {len(articles)} articles on page {page + 1}")
                
                if not articles:
                    print("No more results found.")
                    break
                
                for idx, article in enumerate(articles, 1):
                    try:
                        print(f"\nProcessing article {idx} on page {page + 1}:")
                        
                        # Extract basic information
                        title_elem = article.find("h3", class_="gs_rt")
                        author_elem = article.find("div", class_="gs_a")
                        abstract_elem = article.find("div", class_="gs_rs")
                        
                        # Clean and extract title
                        if title_elem:
                            for span in title_elem.find_all("span"):
                                span.decompose()
                            title = title_elem.get_text(strip=True)
                        else:
                            title = "No title found"
                        
                        # Extract URL and DOI
                        link = title_elem.find("a") if title_elem else None
                        url = link["href"] if link else ""
                        doi = extract_doi(url)
                        
                        # Extract author information
                        author_info = author_elem.get_text(strip=True) if author_elem else ""
                        authors = author_info.split('-')[0].strip() if '-' in author_info else ""
                        
                        # Extract journal
                        journal = author_info.split('-')[-1].strip() if '-' in author_info else ""
                        
                        # Extract year
                        year_match = re.search(r'\b(19|20)\d{2}\b', author_info) if author_info else None
                        year = year_match.group(0) if year_match else ""
                        
                        # Extract abstract
                        abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""
                        
                        # Extract methods and field
                        methods = extract_methods(abstract)
                        field = extract_field(author_info, title, abstract)
                        
                        # Store the result
                        result = {
                            "Title": title,
                            "Authors": authors,
                            "Year": year,
                            "Journal": journal,
                            "DOI": doi,
                            "Field": field,
                            "Methods": methods,
                            "Keywords": keyword,
                            "URL": url,
                            "Abstract": abstract,
                            "Page": page + 1
                        }
                        
                        print(f"Title: {title[:100]}...")
                        print(f"Authors: {authors[:100]}...")
                        print(f"Year: {year}")
                        print(f"Field: {field}")
                        
                        results.append(result)
                        
                    except Exception as e:
                        print(f"Error processing article {idx} on page {page + 1}: {str(e)}")
                        continue
                
            else:
                print(f"Failed to get results for page {page + 1}. Status code: {response.status_code}")
                break
                
        except Exception as e:
            print(f"Error during search on page {page + 1}: {str(e)}")
            break
    
    return results

def main():
    # Configuration
    START_YEAR = 2018
    END_YEAR = 2025
    NUM_PAGES = 3
    
    # Queries
    Keywords = [
        '"digitalsreeni" AND "youtube"',    # AND operator. OR can also be used.
        '"Sreenivas Bhattiprolu"',                       # Exact name search
        '"Sreenivas Bhattiprolu" AND "ZEISS"'       
    ]
    

    
    print(f"\nStarting literature search:")
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Pages per search: {NUM_PAGES}")
    print("Keywords:")
    for keyword in Keywords:
        print(f"- {keyword}")
        
        
    all_results = []
    
    for keyword in Keywords:
        results = search_scholar(keyword, START_YEAR, END_YEAR, num_pages=NUM_PAGES)
        if results:
            all_results.extend(results)
        print(f"\nFound {len(results)} total results for {keyword}")
        
        # Add delay between keywords
        if keyword != Keywords[-1]:  # If not the last keyword
            delay = random.uniform(30, 60)
            print(f"\nWaiting {delay:.1f} seconds before next search...")
            time.sleep(delay)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['Title'])
        final_count = len(df)
        
        print(f"\nFound {initial_count} total results")
        print(f"After removing duplicates: {final_count} results")
        
        # Sort by year
        df = df.sort_values(by=['Year', 'Title'], ascending=[False, True])
        
        # Reorder columns
        columns_order = [
            'Title', 'Authors', 'Year', 'Journal', 'DOI', 'Field', 
            'Methods', 'Keywords', 'URL', 'Abstract', 'Page'
        ]
        df = df[columns_order]
        
        # Save to Excel with current timestamp
        output_file = f'literature_search_{START_YEAR}-{END_YEAR}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        df.to_excel(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Print sample results
        print("\nSample of results:")
        for _, row in df.head().iterrows():
            print("\n" + "="*80)
            print(f"Title: {row['Title']}")
            print(f"Authors: {row['Authors']}")
            print(f"Year: {row['Year']}")
            print(f"Journal: {row['Journal']}")
            print(f"Field: {row['Field']}")
            print(f"DOI: {row['DOI']}")
            print(f"Page: {row['Page']}")
    else:
        print("\nNo results found")

if __name__ == "__main__":
    main()