#!/usr/bin/env python3
"""
Setup script for Enhanced Legal Research Assistant with Memory
"""

import os
import sqlite3
from datetime import datetime

def create_enhanced_requirements():
    """Create updated requirements.txt with additional dependencies"""
    requirements = """flask==2.3.3
flask-socketio==5.3.6
python-dotenv==1.0.0
langchain-google-genai==1.0.8
langchain==0.1.20
requests==2.31.0
beautifulsoup4==4.12.2
aiohttp==3.8.6
asyncio==3.4.3
python-engineio==4.7.1
python-socketio==5.9.0
eventlet==0.33.3
sqlite3
uuid
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements.strip())
    print("✓ Updated requirements.txt with enhanced dependencies")

def initialize_database():
    """Initialize the SQLite database with tables"""
    try:
        conn = sqlite3.connect('research_history.db')
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                paper_content TEXT NOT NULL,
                research_angle TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'completed',
                is_favorite BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Create research_results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                base_arguments TEXT,
                research_analysis TEXT,
                keywords TEXT,
                sources TEXT,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')
        
        # Create user_preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preference_key TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✓ Database initialized successfully")
        
    except Exception as e:
        print(f"✗ Error initializing database: {str(e)}")

def create_sample_data():
    """Create some sample conversations for demonstration"""
    try:
        conn = sqlite3.connect('research_history.db')
        cursor = conn.cursor()
        
        # Check if we already have data
        cursor.execute('SELECT COUNT(*) FROM conversations')
        if cursor.fetchone()[0] > 0:
            print("✓ Sample data already exists")
            conn.close()
            return
        
        # Sample conversations
        sample_conversations = [
            {
                'id': 'sample-1',
                'title': 'AI and Intellectual Property Rights',
                'paper_content': 'This paper examines the intersection of artificial intelligence and intellectual property law...',
                'research_angle': 'Analyzing the impact of AI on intellectual property rights',
                'status': 'completed',
                'is_favorite': True
            },
            {
                'id': 'sample-2',
                'title': 'Privacy Law in Digital Age',
                'paper_content': 'The digital transformation has created new challenges for privacy law...',
                'research_angle': 'Examining privacy protection mechanisms in the digital era',
                'status': 'completed',
                'is_favorite': False
            },
            {
                'id': 'sample-3',
                'title': 'Contract Law and Smart Contracts',
                'paper_content': 'Smart contracts represent a new paradigm in contract law...',
                'research_angle': 'Legal implications of smart contracts in traditional contract law',
                'status': 'completed',
                'is_favorite': True
            }
        ]
        
        for conv in sample_conversations:
            cursor.execute('''
                INSERT INTO conversations (id, title, paper_content, research_angle, status, is_favorite)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (conv['id'], conv['title'], conv['paper_content'], conv['research_angle'], 
                  conv['status'], conv['is_favorite']))
        
        conn.commit()
        conn.close()
        print("✓ Sample conversation data created")
        
    except Exception as e:
        print(f"✗ Error creating sample data: {str(e)}")

def backup_existing_files():
    """Backup existing files before overwriting"""
    files_to_backup = ['app.py', 'templates/index.html', 'static/js/app.js']
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = f"{file_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                os.rename(file_path, backup_path)
                print(f"✓ Backed up {file_path} to {backup_path}")
            except Exception as e:
                print(f"✗ Failed to backup {file_path}: {str(e)}")

def main():
    """Main setup function"""
    print("🚀 Setting up Enhanced Legal Research Assistant with Memory...\n")
    
    # Create backup of existing files
    print("📁 Creating backups of existing files...")
    backup_existing_files()
    print()
    
    # Create updated requirements
    print("📦 Updating requirements...")
    create_enhanced_requirements()
    print()
    
    # Initialize database
    print("🗃️ Setting up database...")
    initialize_database()
    create_sample_data()
    print()
    
    print("✅ Enhanced setup complete!")
    print("\n📋 Next steps:")
    print("1. Replace your existing files with the enhanced versions:")
    print("   - Copy the enhanced app.py")
    print("   - Replace templates/index.html")
    print("   - Replace static/js/app.js")
    print("\n2. Install any new dependencies:")
    print("   pip install -r requirements.txt")
    print("\n3. Run the enhanced application:")
    print("   python app.py")
    print("\n4. Open your browser to:")
    print("   http://localhost:5000")
    
    print("\n🌟 New Features:")
    print("- 💾 Persistent conversation history")
    print("- ⭐ Favorite conversations")
    print("- 🔍 Search through past research")
    print("- 📊 Enhanced results visualization")
    print("- 💾 Auto-save functionality")
    print("- 📤 Export research results")
    print("- 🔗 Share research links")
    print("- 📱 Responsive sidebar design")

if __name__ == "__main__":
    main()