import pandas as pd
import psycopg2
import requests
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urlparse, unquote
import mwparserfromhell
import re

def normalize_wikipedia_url(url):
    """Add https:// if missing from Wikipedia URL."""
    if not url or pd.isna(url):
        return None
    
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    return url

def extract_title_from_url(url):
    """Extract and clean the title from Wikipedia URL."""
    encoded_title = urlparse(url).path.split('/')[-1]
    
    # Decode URL-encoded characters
    decoded_title = unquote(encoded_title)
    
    # Replace underscores with spaces
    cleaned_title = decoded_title.replace('_', ' ')
    
    return cleaned_title

def get_wikipedia_content(url):
    """Fetch and parse Wikipedia page content including tables."""
    try:
        print("Fetching URL:", url)
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the main content area
        content_div = soup.find(id='mw-content-text')
        if not content_div:
            print("Could not find content div with id 'mw-content-text'")
            return None
            
        # Remove only navigation boxes and references
        for unwanted in content_div.find_all(['div'], class_=['navbox', 'reference']):
            unwanted.decompose()
        
        # Process content in order of appearance
        content_parts = []
        
        # Flag to track if we're in a section to skip
        # Track section skipping with level information
        skip_section = False
        skip_section_level = 0  # Track the level of the section we're skipping
        sections_to_skip = ['See also', 'Notes', 'References', 'External links', 'Citations', 'Works cited', 'Bibliography' ]
        
        # Process all content in order of appearance
        for element in content_div.descendants:
            if not isinstance(element, (str, NavigableString)) and element.name:


                # Handle section headings
                if element.name == 'div' and element.get('class') and 'mw-heading' in element.get('class'):
                    heading = element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    if heading:
                        heading_text = heading.get_text().strip()
                        current_level = int(heading.name[1])  # Get current heading level

                        # If we're skipping a section, only stop skipping if we find a heading
                        # at the same or higher level (lower number) as the section we're skipping
                        if skip_section and current_level <= skip_section_level:
                            skip_section = False
                            skip_section_level = 0

                        # Check if this is a section we want to skip
                        if heading_text in sections_to_skip:
                            skip_section = True
                            skip_section_level = current_level
                            continue
                        elif not skip_section:
                            content_parts.append(f"\n{'#' * current_level} {heading_text}\n")
                            continue
                
                # Skip content if we're in a section to skip
                if skip_section:
                    continue

                # Skip elements that are children of lists (they'll be handled by process_list)
                if element.name in ['ul', 'ol'] and any(parent.name in ['ul', 'ol'] for parent in element.parents):
                    continue
                    
                # Skip elements that are children of tables (they'll be handled by process_table)    
                if element.name in ['td', 'th', 'tr', 'caption'] and any(parent.name == 'table' for parent in element.parents):
                    continue

                if element.name == 'p':
                    text = element.get_text().strip()
                    if text:
                        # print(f"Found paragraph: {text[:50]}...")
                        content_parts.append(text)
                        
                elif element.name in ['ul', 'ol']:
                    list_content = process_list(element)
                    if list_content:
                        # print(f"Found list with {len(list_content.split('\\n'))} items")
                        content_parts.append(list_content)
                        
                elif element.name == 'table':
                    table_content = process_table(element)
                    if table_content:
                        # print(f"Found table with {len(table_content.split('\\n'))} rows")
                        content_parts.append(table_content)

        # Process any remaining tables (excluding those in skipped sections)
        if not skip_section:
            tables = content_div.find_all('table')
            for table in tables:
                # Skip tables in Notes/References sections
                if any(parent.name == 'div' and parent.get('class') and 'mw-heading' in parent.get('class') 
                      and any(skip_text in parent.get_text() for skip_text in sections_to_skip) 
                      for parent in table.parents):
                    continue
                    
                table_content = process_table(table)
                if table_content:
                    content_parts.append(table_content)
                else:
                    print("Table processing returned None")
        
        # Join all parts with proper spacing
        if content_parts:
            final_content = '\n\n'.join(filter(None, content_parts))
            # print(f"Final content length: {len(final_content)}")
            return final_content
        else:
            print("No content parts were collected")
            return None
        
    except Exception as e:
        print(f"Error in get_wikipedia_content: {str(e)}")
        return None

def process_table(table):
    """Process a Wikipedia table into a formatted string."""
    try:
        table_data = []
        
        # Extract table caption/title if exists
        caption = table.find('caption')
        if caption:
            table_data.append(f"\nTable: {caption.get_text(strip=True)}\n")
        
        # Get maximum columns by analyzing all rows
        max_cols = 0
        rows = table.find_all('tr')
        # print(f"Processing table with {len(rows)} rows")
        
        for row in rows:
            cols = 0
            for cell in row.find_all(['th', 'td']):
                colspan = int(cell.get('colspan', 1))
                cols += colspan
            max_cols = max(max_cols, cols)
        
        # print(f"Detected {max_cols} columns")
        
        if max_cols == 0:
            print("No columns detected in table")
            return None
        
        # Initialize rowspan tracker
        active_rowspans = []
        for _ in range(max_cols):
            active_rowspans.append({'content': None, 'spans': 0})
        
        # Process all rows
        rows_processed = 0
        for tr in rows:
            row = []
            current_col = 0
            
            # Handle active rowspans
            col_idx = 0
            while col_idx < max_cols:
                span_info = active_rowspans[col_idx]
                if span_info['spans'] > 0:
                    row.append(span_info['content'])
                    span_info['spans'] -= 1
                    col_idx += 1
                else:
                    break
            
            current_col = col_idx
            
            # Process cells in this row
            cells = tr.find_all(['td', 'th'])
            for cell in cells:
                if current_col >= max_cols:
                    break
                
                cell_text = cell.get_text(strip=True)
                colspan = int(cell.get('colspan', 1))
                rowspan = int(cell.get('rowspan', 1))
                
                # Handle colspan
                for i in range(colspan):
                    if current_col + i < max_cols:
                        row.append(cell_text)
                        # Handle rowspan
                        if rowspan > 1:
                            active_rowspans[current_col + i] = {
                                'content': cell_text,
                                'spans': rowspan - 1
                            }
                
                current_col += colspan
            
            # Fill any remaining columns
            while len(row) < max_cols:
                row.append('')
            
            # Add row to table data
            if rows_processed == 0 and any('th' in str(cell) for cell in cells):
                # This is a header row
                if any(cell.strip() for cell in row):
                    table_data.append("| " + " | ".join(row) + " |")
                    table_data.append("|" + "---|" * max_cols)
            else:
                if any(cell.strip() for cell in row):
                    table_data.append("| " + " | ".join(row) + " |")
            
            rows_processed += 1
        
        # Add table footnotes
        footnotes = table.find_all('tr', class_='sortbottom')
        if footnotes:
            for footnote in footnotes:
                footnote_text = footnote.get_text(strip=True)
                if footnote_text:
                    table_data.append(f"\nNote: {footnote_text}")
        
        if table_data:
            return "\n".join(table_data)
        else:
            print("No table data collected")
            return None
            
    except Exception as e:
        print(f"Error processing table: {str(e)}")
        return None

def process_list(list_element, level=0):
    """Process ul/ol elements recursively."""
    try:
        if not list_element:
            return None
        
        list_items = []
        indent = "  " * level
        
        for item in list_element.find_all('li', recursive=False):
            # Get the main text of the list item
            item_text = []
            for content in item.children:
                if not content.name:  # Direct text
                    text = content.strip()
                    if text:
                        item_text.append(text)
                elif content.name in ['ul', 'ol']:  # Nested list
                    nested_list = process_list(content, level + 1)
                    if nested_list:
                        item_text.append('\n' + nested_list)
                else:  # Other elements (spans, links, etc.)
                    text = content.get_text().strip()
                    if text:
                        item_text.append(text)
            
            # Combine the item's text parts
            combined_text = ' '.join(item_text).strip()
            if combined_text:
                prefix = '•' if list_element.name == 'ul' else f"{len(list_items) + 1}."
                list_items.append(f"{indent}{prefix} {combined_text}")
        
        return '\n'.join(list_items) if list_items else None
        
    except Exception as e:
        print(f"Error processing list: {str(e)}")
        return None

def clean_wikicode(raw_content):
    """Clean Wikipedia content using mwparserfromhell, preserving tables and lists."""
    try:
        wikicode = mwparserfromhell.parse(raw_content)
        
        # Filters for references and file/image links only
        re_rm_wikilink = re.compile("^(?:File|Image|Media):", flags=re.IGNORECASE | re.UNICODE)
        
        def rm_wikilink(obj):
            return bool(re_rm_wikilink.match(str(obj.title)))
        
        def rm_tag(obj):
            return str(obj.tag) in {"ref"}  # Only remove references
        
        def rm_template(obj):
            return obj.name.lower() in {
                "reflist", "notelist", "notelist-ua", 
                "notelist-lr", "notelist-ur", "notelist-lg"
            }
        
        def try_remove_obj(obj, section):
            try:
                section.remove(obj)
            except ValueError:
                pass
        
        section_text = []
        # Filter individual sections to clean
        for section in wikicode.get_sections(flat=True, include_lead=True, include_headings=True):
            # Remove unwanted elements
            for obj in section.ifilter_wikilinks(matches=rm_wikilink, recursive=True):
                try_remove_obj(obj, section)
            for obj in section.ifilter_templates(matches=rm_template, recursive=True):
                try_remove_obj(obj, section)
            for obj in section.ifilter_tags(matches=rm_tag, recursive=True):
                try_remove_obj(obj, section)
            
            cleaned_text = section.strip_code().strip()
            if cleaned_text:
                section_text.append(cleaned_text)
        
        return "\n\n".join(section_text)
    except Exception as e:
        print(f"Error cleaning wikicode: {str(e)}")
        return None

def main():
    # Database connection
    conn = psycopg2.connect(
        dbname="frames_new",
        user="postgres",
        password="postgres",
        host="localhost",
        port=5433
    )
    conn.autocommit = False  # Explicitly set autocommit to False
    cur = conn.cursor()
    
    # Create table with additional columns for metadata
    cur.execute("""
        CREATE TABLE IF NOT EXISTS wikipedia_content (
            id SERIAL PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            content TEXT,
            url TEXT,
            processed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()  # Commit table creation
    
    # Load and process data
    df = pd.read_csv('test.tsv', sep='\t')
    
    processed_count = 0
    error_count = 0
    
    try:
        for _, row in df.iterrows():
            # Process each wikipedia_link column
            for i in range(1, 12):
                col_name = f'wikipedia_link_{i}'
                if col_name not in row or pd.isna(row[col_name]):
                    continue
                
                url = normalize_wikipedia_url(row[col_name])
                if not url:
                    continue
                
                try:
                    # Check if URL already exists in database
                    cur.execute("SELECT id FROM wikipedia_content WHERE url = %s", (url,))
                    if cur.fetchone():
                        print(f"Skipping already processed URL: {url}")
                        continue
                    
                    content = get_wikipedia_content(url)
                    if content:
                        title = extract_title_from_url(url)
                        cur.execute(
                            """
                            INSERT INTO wikipedia_content (title, content, url) 
                            VALUES (%s, %s, %s)
                            """,
                            (title, content, url)
                        )
                        conn.commit()  # Commit after each successful insertion
                        processed_count += 1
                        print(f"Successfully processed and committed - Title: {title}")
                    else:
                        print(f'No content found for URL: {url}')
                
                except Exception as row_error:
                    error_count += 1
                    print(f"Error processing URL {url}: {str(row_error)}")
                    conn.rollback()  # Rollback the failed transaction
                    continue  # Continue with the next URL
        
        print(f"\nProcessing completed:")
        print(f"Successfully processed: {processed_count} entries")
        print(f"Errors encountered: {error_count} entries")
        
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()


