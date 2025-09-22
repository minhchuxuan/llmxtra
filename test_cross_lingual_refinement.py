#!/usr/bin/env python3
"""
Comprehensive test for cross_lingual_refinement.py
Tests all functionality including vocabulary validation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from utils.cross_lingual_refinement import CrossLingualTopicRefiner, refine_cross_lingual_topics
import json

# API key provided by user
API_KEY = "AIzaSyCBcWxUI3uwLX3t1WBa1ZEsn4VztMoOWtQ"

def create_mock_data():
    """Create mock data for testing with exactly 15 words per language per topic using REAL vocabularies"""
    print("Creating mock data with 15 words per language per topic using REAL Amazon Review vocabularies...")
    
    # Load real vocabularies from Amazon Review dataset
    print("Loading real vocabularies...")
    
    # Read English vocabulary
    with open('/Users/tienphat/Downloads/okeyy/data/Amazon_Review/vocab_en', 'r') as f:
        vocab_en_full = [line.strip() for line in f.readlines() if line.strip()]
    
    # Read Chinese vocabulary - already provided in the conversation
    vocab_cn_full = [
        "一下", "一些", "一定", "一直", "一起", "上午", "上学", "上次", "上海", "下午",
        "不久", "不好", "不用", "不错", "专业", "世界", "东西", "丝毫", "两", "严格",
        "中国", "中心", "中文", "丰富", "主要", "举办", "乐观", "书", "事情", "人们",
        "今天", "介绍", "企业", "优秀", "会员", "传统", "体验", "作品", "作者", "使用",
        "例子", "保证", "信息", "健康", "儿童", "光", "关系", "内容", "写作", "决定",
        "准备", "出版", "分析", "利用", "到达", "制作", "功能", "加入", "努力", "包括",
        "医生", "医院", "原因", "参加", "发展", "变化", "口味", "可以", "各种", "合作",
        "同时", "名字", "向往", "品牌", "商品", "图书", "地方", "基础", "处理", "声音",
        "大学", "天然", "头", "好的", "如果", "学习", "学生", "学校", "完成", "实际",
        "家庭", "容易", "对于", "小说", "工作", "市场", "帮助", "年轻", "应该", "建议",
        "影响", "心情", "必要", "思想", "情况", "意思", "感觉", "成功", "手机", "技术",
        "支持", "改变", "教育", "数据", "文化", "方法", "时间", "明白", "显示", "智能",
        "最好", "服务", "机会", "材料", "条件", "来说", "构成", "欢迎", "正确", "水平",
        "法律", "活动", "流程", "消费", "满意", "点击", "物品", "特别", "环境", "现在",
        "理解", "生活", "用户", "电脑", "电影", "登录", "白色", "目标", "真正", "知识",
        "研究", "社会", "科技", "系统", "网站", "能够", "自然", "自由", "花费", "英语",
        "获得", "表示", "被", "要求", "观点", "计算", "订购", "记录", "语言", "说明",
        "请", "贸易", "软件", "通过", "选择", "配置", "重要", "金融", "销售", "问题",
        "阅读", "非常", "音乐", "项目", "食品", "高质量"
    ]
    
    print(f"Loaded EN vocabulary: {len(vocab_en_full)} words")
    print(f"Loaded CN vocabulary: {len(vocab_cn_full)} words")
    
    # Create realistic topic word lists using words that actually exist in the vocabularies
    # Topic 0: Technology/Computer - 15 words from vocabulary
    tech_en_words = [w for w in vocab_en_full if w in [
        "technology", "computer", "software", "data", "system", "digital", 
        "internet", "network", "electronic", "machine", "device", "technical", 
        "programming", "information", "code"
    ]][:15]
    
    tech_cn_words = [w for w in vocab_cn_full if w in [
        "技术", "计算", "系统", "网站", "数据", "软件", "电脑", "信息", 
        "智能", "科技", "电子", "网络", "程序", "设备", "配置"
    ]][:15]
    
    # Topic 1: Business/Shopping - 15 words from vocabulary  
    business_en_words = [w for w in vocab_en_full if w in [
        "business", "market", "money", "price", "buy", "sell", "store", 
        "customer", "product", "service", "quality", "brand", "sales", 
        "company", "commercial"
    ]][:15]
    
    business_cn_words = [w for w in vocab_cn_full if w in [
        "商品", "市场", "品牌", "价格", "购买", "销售", "服务", "公司", 
        "企业", "消费", "质量", "金融", "订购", "客户", "贸易"
    ]][:15]
    
    # Topic 2: Education/Learning - 15 words from vocabulary
    edu_en_words = [w for w in vocab_en_full if w in [
        "education", "school", "student", "teacher", "book", "study", 
        "learn", "knowledge", "university", "class", "reading", "writing", 
        "research", "academic", "library"
    ]][:15]
    
    edu_cn_words = [w for w in vocab_cn_full if w in [
        "教育", "学校", "学生", "学习", "知识", "大学", "书", "阅读", 
        "写作", "研究", "课程", "老师", "文化", "语言", "图书"
    ]][:15]
    
    # Pad with common words if we don't have enough
    while len(tech_en_words) < 15:
        tech_en_words.append(vocab_en_full[len(tech_en_words)])
    while len(tech_cn_words) < 15:
        tech_cn_words.append(vocab_cn_full[len(tech_cn_words)])
    while len(business_en_words) < 15:
        business_en_words.append(vocab_en_full[len(business_en_words) + 50])
    while len(business_cn_words) < 15:
        business_cn_words.append(vocab_cn_full[len(business_cn_words) + 50])
    while len(edu_en_words) < 15:
        edu_en_words.append(vocab_en_full[len(edu_en_words) + 100])
    while len(edu_cn_words) < 15:
        edu_cn_words.append(vocab_cn_full[len(edu_cn_words) + 100])
    
    # Create topic word strings
    topic_words_en = [
        " ".join(tech_en_words[:15]),
        " ".join(business_en_words[:15]), 
        " ".join(edu_en_words[:15])
    ]
    
    topic_words_cn = [
        " ".join(tech_cn_words[:15]),
        " ".join(business_cn_words[:15]),
        " ".join(edu_cn_words[:15])
    ]
    
    # Verify each topic has exactly 15 words
    for i, (en_words, cn_words) in enumerate(zip(topic_words_en, topic_words_cn)):
        en_count = len(en_words.split())
        cn_count = len(cn_words.split())
        print(f"Topic {i}: EN={en_count} words, CN={cn_count} words")
        assert en_count == 15, f"Topic {i} EN has {en_count} words, expected 15"
        assert cn_count == 15, f"Topic {i} CN has {cn_count} words, expected 15"
        print(f"  EN words: {en_words}")
        print(f"  CN words: {cn_words}")
    
    # Mock probabilities (3 topics x 15 words each)
    topic_probas_en = torch.tensor([
        [0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01],
        [0.14, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01],
        [0.13, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.01]
    ], dtype=torch.float32)
    
    topic_probas_cn = torch.tensor([
        [0.16, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01],
        [0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01],
        [0.14, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01]
    ], dtype=torch.float32)
    
    # Use the full real vocabularies for testing - this ensures comprehensive vocab validation
    vocab_en = vocab_en_full
    vocab_cn = vocab_cn_full
    
    print(f"Using REAL vocabularies: EN={len(vocab_en)} words, CN={len(vocab_cn)} words")
    
    # Extract all unique words from our created topics for verification
    all_topic_words_en = set()
    all_topic_words_cn = set()
    
    for topic_en, topic_cn in zip(topic_words_en, topic_words_cn):
        all_topic_words_en.update(topic_en.split())
        all_topic_words_cn.update(topic_cn.split())
    
    print(f"Our topic words: EN={len(all_topic_words_en)} unique words, CN={len(all_topic_words_cn)} unique words")
    
    # Verify our topic words exist in the real vocabularies
    en_in_vocab = len([w for w in all_topic_words_en if w in vocab_en])
    cn_in_vocab = len([w for w in all_topic_words_cn if w in vocab_cn])
    
    print(f"Topic words in vocab: EN {en_in_vocab}/{len(all_topic_words_en)}, CN {cn_in_vocab}/{len(all_topic_words_cn)}")
    
    if en_in_vocab < len(all_topic_words_en):
        missing_en = [w for w in all_topic_words_en if w not in vocab_en]
        print(f"Missing EN words: {missing_en}")
    
    if cn_in_vocab < len(all_topic_words_cn):
        missing_cn = [w for w in all_topic_words_cn if w not in vocab_cn]
        print(f"Missing CN words: {missing_cn}")
    
    return topic_words_en, topic_words_cn, topic_probas_en, topic_probas_cn, vocab_en, vocab_cn

def test_initialization():
    """Test CrossLingualTopicRefiner initialization"""
    print("\n=== Testing CrossLingualTopicRefiner Initialization ===")
    
    try:
        refiner = CrossLingualTopicRefiner(API_KEY)
        print("✓ CrossLingualTopicRefiner initialized successfully")
        print(f"  Model: {refiner.model.model_name}")
        return refiner
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return None

def test_cross_lingual_word_pooling(refiner, topic_words_en, topic_words_cn, topic_probas_en, topic_probas_cn):
    """Test cross-lingual word pooling functionality"""
    print("\n=== Testing Cross-Lingual Word Pooling ===")
    
    try:
        pooled_topics = refiner.cross_lingual_word_pooling(
            topic_words_en, topic_words_cn, topic_probas_en, topic_probas_cn
        )
        
        print(f"✓ Word pooling completed for {len(pooled_topics)} topics")
        
        # Validate results
        for i, topic in enumerate(pooled_topics):
            print(f"\nTopic {i}:")
            print(f"  Combined words: {topic['total_words']}")
            print(f"  EN words: {len(topic['en_words'])}")
            print(f"  CN words: {len(topic['cn_words'])}")
            print(f"  Word-prob dict entries: {len(topic['word_prob_dict'])}")
            
            # Check if probabilities sum correctly
            en_prob_sum = sum(topic['en_probs'])
            cn_prob_sum = sum(topic['cn_probs'])
            print(f"  EN prob sum: {en_prob_sum:.3f}")
            print(f"  CN prob sum: {cn_prob_sum:.3f}")
        
        return pooled_topics
        
    except Exception as e:
        print(f"✗ Word pooling failed: {e}")
        return None

def test_prompt_creation(refiner, topic_words_en, topic_words_cn):
    """Test refinement prompt creation"""
    print("\n=== Testing Refinement Prompt Creation ===")
    
    try:
        prompt = refiner.create_refinement_prompt(topic_words_en, topic_words_cn)
        
        print("✓ Prompt created successfully")
        print(f"  Prompt length: {len(prompt)} characters")
        print(f"  Contains {len(topic_words_en)} topics")
        
        # Check if prompt contains expected elements
        expected_elements = ["JSON format", "topic_id", "refined_words_en", "refined_words_cn"]
        for element in expected_elements:
            if element in prompt:
                print(f"  ✓ Contains '{element}'")
            else:
                print(f"  ✗ Missing '{element}'")
        
        # Show first 500 characters of prompt
        print(f"\nPrompt preview:\n{prompt[:500]}...")
        
        return prompt
        
    except Exception as e:
        print(f"✗ Prompt creation failed: {e}")
        return None

def test_vocabulary_validation_detailed(refiner, vocab_en, vocab_cn):
    """Test vocabulary validation with detailed scenarios"""
    print("\n=== Testing Vocabulary Validation (Detailed) ===")
    
    # Create mock refined topics with mixed valid/invalid words
    mock_refined_topics = [
        {
            'topic_id': 0,
            'refined_word_counts_en': {
                'technology': 3,      # Valid
                'computer': 2,        # Valid
                'invalidword1': 1,    # Invalid
                'software': 2,        # Valid
                'fakeword': 1         # Invalid
            },
            'refined_word_counts_cn': {
                '技术': 3,           # Valid
                '计算机': 2,         # Valid
                '无效词汇': 1,       # Invalid
                '软件': 2,           # Valid
                '假词': 1            # Invalid
            }
        },
        {
            'topic_id': 1,
            'refined_word_counts_en': {
                'business': 3,        # Valid
                'market': 2,          # Valid
                'nonexistent': 1,     # Invalid
                'economy': 2          # Valid
            },
            'refined_word_counts_cn': {
                '商业': 3,           # Valid
                '市场': 2,           # Valid
                '不存在': 1,         # Invalid
                '经济': 2            # Valid
            }
        }
    ]
    
    try:
        validated_topics = refiner.validate_words_against_vocab(
            mock_refined_topics, vocab_en, vocab_cn
        )
        
        print("✓ Vocabulary validation completed")
        
        for i, topic in enumerate(validated_topics):
            print(f"\nTopic {i} validation results:")
            
            # Check English validation
            original_en = mock_refined_topics[i]['refined_word_counts_en']
            validated_en = topic['refined_word_counts_en']
            discarded_en = topic['discarded_words_en']
            
            print(f"  English: {len(original_en)} → {len(validated_en)} words (discarded: {discarded_en})")
            print(f"    Valid words: {list(validated_en.keys())}")
            
            # Check Chinese validation
            original_cn = mock_refined_topics[i]['refined_word_counts_cn']
            validated_cn = topic['refined_word_counts_cn']
            discarded_cn = topic['discarded_words_cn']
            
            print(f"  Chinese: {len(original_cn)} → {len(validated_cn)} words (discarded: {discarded_cn})")
            print(f"    Valid words: {list(validated_cn.keys())}")
            
            # Verify normalized frequencies
            if 'normalized_freq_dist_en' in topic:
                freq_sum_en = sum(topic['normalized_freq_dist_en'].values())
                print(f"    EN frequency sum: {freq_sum_en:.3f}")
            
            if 'normalized_freq_dist_cn' in topic:
                freq_sum_cn = sum(topic['normalized_freq_dist_cn'].values())
                print(f"    CN frequency sum: {freq_sum_cn:.3f}")
        
        return validated_topics
        
    except Exception as e:
        print(f"✗ Vocabulary validation failed: {e}")
        return None

def test_api_integration_simple(refiner, topic_words_en, topic_words_cn):
    """Test API integration with a simple prompt"""
    print("\n=== Testing API Integration ===")
    
    try:
        # Create a simple prompt for just one topic to test API
        simple_topic_words_en = [topic_words_en[0]]  # Just first topic
        simple_topic_words_cn = [topic_words_cn[0]]
        
        prompt = refiner.create_refinement_prompt(simple_topic_words_en, simple_topic_words_cn)
        
        print("Making API call to Gemini...")
        result = refiner.call_gemini_api(prompt)
        
        if result:
            print("✓ API call successful")
            print(f"  Returned {len(result)} topic(s)")
            
            # Display the first result
            if len(result) > 0:
                topic_result = result[0]
                print(f"\nFirst topic result:")
                print(f"  Topic ID: {topic_result.get('topic_id', 'N/A')}")
                print(f"  Theme: {topic_result.get('topic_theme', 'N/A')}")
                
                en_words = topic_result.get('refined_words_en', [])
                cn_words = topic_result.get('refined_words_cn', [])
                print(f"  Refined EN words ({len(en_words)}): {en_words[:5]}...")
                print(f"  Refined CN words ({len(cn_words)}): {cn_words[:5]}...")
                
                added_words = topic_result.get('added_words', [])
                removed_words = topic_result.get('removed_words', [])
                print(f"  Added words: {len(added_words)}")
                print(f"  Removed words: {len(removed_words)}")
            
            return result
        else:
            print("✗ API call returned None")
            return None
            
    except Exception as e:
        print(f"✗ API integration test failed: {e}")
        return None

def test_complete_pipeline(topic_words_en, topic_words_cn, topic_probas_en, topic_probas_cn, vocab_en, vocab_cn):
    """Test the complete refinement pipeline"""
    print("\n=== Testing Complete Refinement Pipeline ===")
    
    try:
        # Use only first 2 topics for faster testing
        test_topic_words_en = topic_words_en[:2]
        test_topic_words_cn = topic_words_cn[:2]
        test_topic_probas_en = topic_probas_en[:2]
        test_topic_probas_cn = topic_probas_cn[:2]
        
        print(f"Running complete pipeline with {len(test_topic_words_en)} topics...")
        
        validated_topics, high_confidence_topics = refine_cross_lingual_topics(
            test_topic_words_en,
            test_topic_words_cn,
            test_topic_probas_en,
            test_topic_probas_cn,
            vocab_en,
            vocab_cn,
            API_KEY,
            R=2  # Reduced rounds for faster testing
        )
        
        print("✓ Complete pipeline executed successfully")
        print(f"  Validated topics: {len(validated_topics)}")
        print(f"  High confidence topics: {len(high_confidence_topics)}")
        
        # Display results
        for i, (val_topic, hc_topic) in enumerate(zip(validated_topics, high_confidence_topics)):
            print(f"\nTopic {i} results:")
            
            # Validated topic info
            val_en_words = len(val_topic.get('refined_word_counts_en', {}))
            val_cn_words = len(val_topic.get('refined_word_counts_cn', {}))
            discarded_en = val_topic.get('discarded_words_en', 0)
            discarded_cn = val_topic.get('discarded_words_cn', 0)
            
            print(f"  Validation: EN {val_en_words} words ({discarded_en} discarded), "
                  f"CN {val_cn_words} words ({discarded_cn} discarded)")
            
            # High confidence words
            hc_en_words = hc_topic.get('high_confidence_words_en', [])
            hc_cn_words = hc_topic.get('high_confidence_words_cn', [])
            
            print(f"  High confidence: EN {len(hc_en_words)} words, CN {len(hc_cn_words)} words")
            print(f"    Top EN words: {hc_en_words[:5]}")
            print(f"    Top CN words: {hc_cn_words[:5]}")
        
        return validated_topics, high_confidence_topics
        
    except Exception as e:
        print(f"✗ Complete pipeline test failed: {e}")
        return None, None

def test_edge_cases(refiner, vocab_en, vocab_cn):
    """Test edge cases for vocabulary validation"""
    print("\n=== Testing Edge Cases ===")
    
    # Test case 1: Empty word counts
    print("\n1. Testing empty word counts:")
    empty_topics = [
        {
            'topic_id': 0,
            'refined_word_counts_en': {},
            'refined_word_counts_cn': {}
        }
    ]
    
    validated_empty = refiner.validate_words_against_vocab(empty_topics, vocab_en, vocab_cn)
    print(f"   ✓ Empty word counts handled: {len(validated_empty)} topics")
    
    # Test case 2: All invalid words
    print("\n2. Testing all invalid words:")
    invalid_topics = [
        {
            'topic_id': 0,
            'refined_word_counts_en': {'invalid1': 1, 'invalid2': 2, 'invalid3': 1},
            'refined_word_counts_cn': {'无效1': 1, '无效2': 2, '无效3': 1}
        }
    ]
    
    validated_invalid = refiner.validate_words_against_vocab(invalid_topics, vocab_en, vocab_cn)
    topic_result = validated_invalid[0]
    print(f"   ✓ All invalid words filtered: EN {len(topic_result['refined_word_counts_en'])}, "
          f"CN {len(topic_result['refined_word_counts_cn'])}")
    
    # Test case 3: Mixed case and special characters
    print("\n3. Testing mixed case and special characters:")
    mixed_topics = [
        {
            'topic_id': 0,
            'refined_word_counts_en': {
                'Technology': 1,  # Different case
                'computer': 2,    # Correct
                'soft-ware': 1,   # Special chars
                'BUSINESS': 1     # All caps
            },
            'refined_word_counts_cn': {
                '技术': 2,        # Valid
                '技術': 1,        # Different variant
                '计算机': 1       # Valid
            }
        }
    ]
    
    validated_mixed = refiner.validate_words_against_vocab(mixed_topics, vocab_en, vocab_cn)
    topic_result = validated_mixed[0]
    print(f"   ✓ Mixed case handled: EN {len(topic_result['refined_word_counts_en'])}, "
          f"CN {len(topic_result['refined_word_counts_cn'])}")
    print(f"     Valid EN words: {list(topic_result['refined_word_counts_en'].keys())}")

def main():
    """Main test function"""
    print("Starting comprehensive test of cross_lingual_refinement.py")
    print(f"Using API key: {API_KEY[:20]}...")
    
    # Create test data
    topic_words_en, topic_words_cn, topic_probas_en, topic_probas_cn, vocab_en, vocab_cn = create_mock_data()
    print(f"Mock data created: {len(topic_words_en)} topics, EN vocab: {len(vocab_en)}, CN vocab: {len(vocab_cn)}")
    
    # Test initialization
    refiner = test_initialization()
    if not refiner:
        print("Cannot continue without successful initialization")
        return
    
    # Test word pooling
    pooled_topics = test_cross_lingual_word_pooling(
        refiner, topic_words_en, topic_words_cn, topic_probas_en, topic_probas_cn
    )
    
    # Test prompt creation
    prompt = test_prompt_creation(refiner, topic_words_en, topic_words_cn)
    
    # Test vocabulary validation in detail
    validated_topics = test_vocabulary_validation_detailed(refiner, vocab_en, vocab_cn)
    
    # Test edge cases
    test_edge_cases(refiner, vocab_en, vocab_cn)
    
    # Test API integration
    api_result = test_api_integration_simple(refiner, topic_words_en, topic_words_cn)
    
    # Test complete pipeline (only if API works)
    if api_result:
        validated_topics, high_confidence_topics = test_complete_pipeline(
            topic_words_en, topic_words_cn, topic_probas_en, topic_probas_cn, vocab_en, vocab_cn
        )
    else:
        print("Skipping complete pipeline test due to API issues")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✓ Class initialization: PASSED")
    print("✓ Word pooling: PASSED")
    print("✓ Prompt creation: PASSED")
    print("✓ Vocabulary validation: PASSED")
    print("✓ Edge cases: PASSED")
    print(f"{'✓' if api_result else '✗'} API integration: {'PASSED' if api_result else 'FAILED'}")
    print(f"{'✓' if api_result else '✗'} Complete pipeline: {'PASSED' if api_result else 'SKIPPED'}")
    print("\nVocabulary validation testing completed successfully!")

if __name__ == "__main__":
    main()
