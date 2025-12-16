import React, { useState } from 'react';
import ScrollableBox from './ScrollableBox';

const BioGPTComponent = () => {
    const [inputText, setInputText] = useState('');
    const [processedText, setProcessedText] = useState('');
    const [entitiesByCategory, setEntitiesByCategory] = useState({});
    const [loading, setLoading] = useState(false);
    const [selectedEntity, setSelectedEntity] = useState(null);
    const [bioGPTResponse, setBioGPTResponse] = useState('');
    const [abstractOnly, setAbstractOnly] = useState(true);
    const [meshTerms, setMeshTerms] = useState([]);

    const handleTextChange = (event) => {
        setInputText(event.target.value);
        setProcessedText('');
        setEntitiesByCategory({});
        setBioGPTResponse('');
        setMeshTerms([]);
    };

    // Read file as text and put into input area (accepts .txt, .md)
    const handleFileSelect = (event) => {
        const file = event.target.files && event.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            setInputText(String(e.target.result));
        };
        reader.onerror = () => {
            console.error('File read error');
        };
        reader.readAsText(file);
    };

    const handleProcess = async () => {
        if (!inputText) return;
        setLoading(true);
        try {
            const resp = await fetch('http://localhost:5001/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: inputText, max_new_tokens: 200 }),
            });
            const j = await resp.json();
            if (resp.ok) setProcessedText(j.result || '');
            else {
                console.error('biogpt error', j);
                setProcessedText('Error generating text — check server logs or backend.');
            }
        } catch (err) {
            console.error(err);
            setProcessedText('Error contacting server.');
        } finally {
            setLoading(false);
        }
    };

    const handleExtractKeywords = async () => {
        if (!inputText) return;
        setLoading(true);
        try {
            const r = await fetch('http://localhost:5001/ner', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: inputText, extract_abstract_only: abstractOnly }),
            });
            const j = await r.json();
            if (r.ok) {
                const entityList = j.entities || [];
                
                // Group entities by category
                const grouped = {};
                entityList.forEach(entity => {
                    const category = entity.category || entity.label || 'Other';
                    if (!grouped[category]) {
                        grouped[category] = [];
                    }
                    grouped[category].push(entity);
                });
                setEntitiesByCategory(grouped);
            } else {
                console.error('NER error', j);
            }
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleExtractMesh = async () => {
        if (!inputText) return;
        setLoading(true);
        try {
            const r = await fetch('http://localhost:5001/mesh_extract', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: inputText, extract_abstract_only: abstractOnly }),
            });
            const j = await r.json();
            if (r.ok) {
                setMeshTerms(j.mesh_terms || []);
            } else {
                console.error('MeSH extraction error', j);
            }
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleEntityClick = async (entityText) => {
        setSelectedEntity(entityText);
        setBioGPTResponse('');
        setLoading(true);
        try {
            const prompt = `Provide a concise biomedical description of "${entityText}" (1-3 sentences) and mention common contexts where it appears.`;
            const r = await fetch('http://localhost:5001/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: prompt, max_new_tokens: 120 }),
            });
            const j = await r.json();
            if (r.ok) setBioGPTResponse(j.result || '');
            else console.error('Entity query error', j);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            width: '100%', 
            minHeight: '100vh',
            padding: '40px 20px', 
            boxSizing: 'border-box', 
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
            backgroundColor: '#f8f9fa'
        }}>
            {/* Header */}
            <div style={{ 
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                width: '100%', 
                maxWidth: '1400px',
                padding: '24px 32px',
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center', 
                borderRadius: '2px', 
                marginBottom: '32px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                borderLeft: '4px solid #764ba2'
            }}>
                <h1 style={{ 
                    color: 'white', 
                    margin: 0,
                    fontSize: '28px',
                    fontWeight: '600',
                    letterSpacing: '-0.5px'
                }}>Biomedical Text Analysis (BioGPT + NER + LLaMA)</h1>
                
                <label style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    color: 'white',
                    fontSize: '14px',
                    cursor: 'pointer',
                    userSelect: 'none'
                }}>
                    <input 
                        type="checkbox" 
                        checked={abstractOnly}
                        onChange={(e) => setAbstractOnly(e.target.checked)}
                        style={{ marginRight: '8px', cursor: 'pointer' }}
                    />
                    Extract from abstract only
                </label>
            </div>

            {/* Main Content */}
            <div style={{ 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'flex-start', 
                minHeight: '70vh',
                flexDirection: 'row', 
                gap: '24px', 
                width: '100%',
                maxWidth: '1400px'
            }}>
                {/* Left Panel: Input */}
                <div style={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    backgroundColor: 'white',
                    width: '35%', 
                    minHeight: '600px',
                    padding: '24px', 
                    borderRadius: '2px', 
                    boxSizing: 'border-box', 
                    position: 'relative',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.12)',
                    border: '1px solid #dee2e6'
                }}>
                    <h3 style={{ 
                        margin: '0 0 20px 0', 
                        fontSize: '18px', 
                        fontWeight: '600',
                        color: '#212529',
                        borderBottom: '2px solid #667eea',
                        paddingBottom: '8px'
                    }}>Input Text</h3>
                    
                    <label style={{
                        display: 'inline-block',
                        padding: '10px 18px',
                        backgroundColor: '#28a745',
                        color: 'white',
                        borderRadius: '2px',
                        cursor: 'pointer',
                        textAlign: 'center',
                        fontWeight: '500',
                        fontSize: '14px',
                        marginBottom: '16px',
                        transition: 'background-color 0.2s',
                        border: 'none'
                    }}
                    onMouseEnter={(e) => e.target.style.backgroundColor = '#218838'}
                    onMouseLeave={(e) => e.target.style.backgroundColor = '#28a745'}
                    >
                        Upload File (PDF, TXT, MD)
                        <input type="file" accept=".txt,.md,.pdf" onChange={async (e) => {
                            const file = e.target.files && e.target.files[0];
                            if (!file) return;
                            const ext = file.name.split('.').pop().toLowerCase();
                            if (ext === 'pdf') {
                                setLoading(true);
                                try {
                                    const form = new FormData();
                                    form.append('file', file);
                                    const resp = await fetch('http://localhost:5001/process_file', { method: 'POST', body: form });
                                    const j = await resp.json();
                                    if (resp.ok) {
                                        setInputText(j.text || '');
                                        setProcessedText(j.summary || '');
                                        const entityList = j.entities || [];
                                        // Group by category
                                        const grouped = {};
                                        entityList.forEach(entity => {
                                            const cat = entity.category || entity.label || 'Other';
                                            if (!grouped[cat]) grouped[cat] = [];
                                            grouped[cat].push(entity);
                                        });
                                        setEntitiesByCategory(grouped);
                                    } else {
                                        console.error('process_file error', j);
                                    }
                                } catch (err) {
                                    console.error(err);
                                } finally {
                                    setLoading(false);
                                }
                            } else {
                                handleFileSelect(e);
                            }
                        }} style={{ display: 'none' }} />
                    </label>
                    
                    <textarea 
                        style={{ 
                            width: '100%', 
                            flex: 1,
                            minHeight: '300px',
                            marginBottom: '20px', 
                            padding: '12px', 
                            borderRadius: '2px', 
                            border: '1px solid #ced4da', 
                            resize: 'vertical',
                            fontFamily: 'inherit',
                            fontSize: '14px',
                            lineHeight: '1.6',
                            transition: 'border-color 0.2s',
                            outline: 'none'
                        }} 
                        placeholder="Enter or paste your scientific article text here..."
                        value={inputText} 
                        onChange={handleTextChange}
                        onFocus={(e) => e.target.style.borderColor = '#28a745'}
                        onBlur={(e) => e.target.style.borderColor = '#ced4da'}
                    />
                    
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: 'auto' }}>
                        <div style={{ display: 'flex', gap: '12px' }}>
                            <button 
                                onClick={handleProcess} 
                                disabled={!inputText || loading}
                                style={{
                                    flex: 1,
                                    padding: '12px 20px',
                                    backgroundColor: inputText && !loading ? '#28a745' : '#e9ecef',
                                    color: inputText && !loading ? 'white' : '#6c757d',
                                    border: 'none',
                                    borderRadius: '2px',
                                    fontWeight: '600',
                                    fontSize: '14px',
                                    cursor: inputText && !loading ? 'pointer' : 'not-allowed',
                                    transition: 'background-color 0.2s'
                                }}
                                onMouseEnter={(e) => {
                                    if (inputText && !loading) e.target.style.backgroundColor = '#218838';
                                }}
                                onMouseLeave={(e) => {
                                    if (inputText && !loading) e.target.style.backgroundColor = '#28a745';
                                }}
                            >
                                Analyze Text
                            </button>
                            <button 
                                onClick={handleExtractKeywords} 
                                disabled={!inputText || loading}
                                style={{
                                    flex: 1,
                                    padding: '12px 20px',
                                    backgroundColor: inputText && !loading ? '#fd7e14' : '#e9ecef',
                                    color: inputText && !loading ? 'white' : '#6c757d',
                                    border: 'none',
                                    borderRadius: '2px',
                                    fontWeight: '600',
                                    fontSize: '14px',
                                    cursor: inputText && !loading ? 'pointer' : 'not-allowed',
                                    transition: 'background-color 0.2s'
                                }}
                                onMouseEnter={(e) => {
                                    if (inputText && !loading) e.target.style.backgroundColor = '#e8590c';
                                }}
                                onMouseLeave={(e) => {
                                    if (inputText && !loading) e.target.style.backgroundColor = '#fd7e14';
                                }}
                            >
                                Extract Entities
                            </button>
                        </div>
                        <button 
                            onClick={handleExtractMesh} 
                            disabled={!inputText || loading}
                            style={{
                                width: '100%',
                                padding: '12px 20px',
                                backgroundColor: inputText && !loading ? '#667eea' : '#e9ecef',
                                color: inputText && !loading ? 'white' : '#6c757d',
                                border: 'none',
                                borderRadius: '2px',
                                fontWeight: '600',
                                fontSize: '14px',
                                cursor: inputText && !loading ? 'pointer' : 'not-allowed',
                                transition: 'background-color 0.2s'
                            }}
                            onMouseEnter={(e) => {
                                if (inputText && !loading) e.target.style.backgroundColor = '#5568d3';
                            }}
                            onMouseLeave={(e) => {
                                if (inputText && !loading) e.target.style.backgroundColor = '#667eea';
                            }}
                        >
                            Extract MeSH Terms (LLaMA)
                        </button>
                    </div>
                </div>

                {/* Right Panel: Results */}
                <div style={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    backgroundColor: 'white',
                    width: '65%', 
                    minHeight: '600px',
                    padding: '24px', 
                    borderRadius: '2px', 
                    boxSizing: 'border-box', 
                    overflowY: 'auto',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.12)',
                    border: '1px solid #dee2e6'
                }}>
                    <h3 style={{ 
                        margin: '0 0 20px 0', 
                        fontSize: '18px', 
                        fontWeight: '600',
                        color: '#212529',
                        borderBottom: '2px solid #667eea',
                        paddingBottom: '8px'
                    }}>Analysis Results</h3>
                    
                    {loading ? (
                        <div style={{ 
                            display: 'flex', 
                            flexDirection: 'column',
                            alignItems: 'center', 
                            justifyContent: 'center',
                            padding: '60px 20px',
                            color: '#667eea'
                        }}>
                            <div style={{
                                width: '40px',
                                height: '40px',
                                border: '3px solid #e9ecef',
                                borderTopColor: '#667eea',
                                borderRadius: '50%',
                                animation: 'spin 0.8s linear infinite',
                                marginBottom: '16px'
                            }}></div>
                            <p style={{ fontSize: '14px', fontWeight: '500', color: '#6c757d' }}>Processing...</p>
                            <style>{`
                                @keyframes spin {
                                    to { transform: rotate(360deg); }
                                }
                            `}</style>
                        </div>
                    ) : (
                        <>
                            {processedText && (
                                <div style={{ 
                                    border: '1px solid #dee2e6', 
                                    padding: '16px', 
                                    borderRadius: '2px', 
                                    marginBottom: '20px', 
                                    backgroundColor: '#f8f9fa'
                                }}>
                                    <h4 style={{ 
                                        margin: '0 0 12px 0', 
                                        fontSize: '16px', 
                                        fontWeight: '600',
                                        color: '#212529'
                                    }}>Generated Summary</h4>
                                    <div style={{ 
                                        fontSize: '14px', 
                                        lineHeight: '1.7',
                                        color: '#495057'
                                    }}>
                                        <ScrollableBox title={processedText} />
                                    </div>
                                </div>
                            )}

                            {meshTerms.length > 0 && (
                                <div style={{ 
                                    border: '1px solid #dee2e6', 
                                    padding: '16px', 
                                    borderRadius: '2px', 
                                    marginBottom: '20px', 
                                    backgroundColor: '#f8f9fa',
                                    borderLeft: '4px solid #667eea'
                                }}>
                                    <h4 style={{ 
                                        margin: '0 0 12px 0', 
                                        fontSize: '16px', 
                                        fontWeight: '600',
                                        color: '#212529'
                                    }}>MeSH Terms (Fine-tuned LLaMA 3.1-8B)</h4>
                                    <div style={{ 
                                        display: 'flex', 
                                        flexWrap: 'wrap', 
                                        gap: '8px',
                                        marginTop: '12px'
                                    }}>
                                        {meshTerms.map((term, idx) => (
                                            <div 
                                                key={idx}
                                                onClick={() => handleEntityClick(term)}
                                                style={{ 
                                                    backgroundColor: selectedEntity === term ? '#667eea' : '#ffffff', 
                                                    color: selectedEntity === term ? 'white' : '#495057', 
                                                    padding: '6px 14px', 
                                                    borderRadius: '2px', 
                                                    fontWeight: '500',
                                                    fontSize: '13px',
                                                    cursor: 'pointer',
                                                    transition: 'all 0.15s',
                                                    border: selectedEntity === term ? '1px solid #667eea' : '1px solid #ced4da'
                                                }}
                                                onMouseEnter={(e) => {
                                                    if (selectedEntity !== term) {
                                                        e.target.style.backgroundColor = '#f8f9fa';
                                                        e.target.style.borderColor = '#667eea';
                                                    }
                                                }}
                                                onMouseLeave={(e) => {
                                                    if (selectedEntity !== term) {
                                                        e.target.style.backgroundColor = '#ffffff';
                                                        e.target.style.borderColor = '#ced4da';
                                                    }
                                                }}
                                            >
                                                {term}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {Object.keys(entitiesByCategory).length > 0 && (
                                <div style={{ marginBottom: '20px' }}>
                                    <h4 style={{ 
                                        margin: '0 0 16px 0', 
                                        fontSize: '16px', 
                                        fontWeight: '600',
                                        color: '#212529'
                                    }}>Extracted Entities by Category (Biomedical NER)</h4>
                                    {Object.entries(entitiesByCategory).map(([category, entityList]) => (
                                        <div key={category} style={{ marginBottom: '20px' }}>
                                            <div style={{
                                                fontSize: '13px',
                                                fontWeight: '600',
                                                color: '#6c757d',
                                                textTransform: 'uppercase',
                                                letterSpacing: '0.5px',
                                                marginBottom: '8px',
                                                paddingBottom: '4px',
                                                borderBottom: '1px solid #dee2e6'
                                            }}>
                                                {category}
                                            </div>
                                            <div style={{ 
                                                display: 'flex', 
                                                flexWrap: 'wrap', 
                                                gap: '8px',
                                                marginTop: '8px'
                                            }}>
                                                {entityList.map((entity, idx) => (
                                                    <div 
                                                        key={idx} 
                                                        onClick={() => handleEntityClick(entity.text)} 
                                                        style={{ 
                                                            backgroundColor: selectedEntity === entity.text ? '#667eea' : '#ffffff', 
                                                            color: selectedEntity === entity.text ? 'white' : '#495057', 
                                                            padding: '6px 14px', 
                                                            borderRadius: '2px', 
                                                            fontWeight: '500',
                                                            fontSize: '13px',
                                                            cursor: 'pointer',
                                                            transition: 'all 0.15s',
                                                            border: selectedEntity === entity.text ? '1px solid #667eea' : '1px solid #ced4da'
                                                        }}
                                                        onMouseEnter={(e) => {
                                                            if (selectedEntity !== entity.text) {
                                                                e.target.style.backgroundColor = '#f8f9fa';
                                                                e.target.style.borderColor = '#667eea';
                                                            }
                                                        }}
                                                        onMouseLeave={(e) => {
                                                            if (selectedEntity !== entity.text) {
                                                                e.target.style.backgroundColor = '#ffffff';
                                                                e.target.style.borderColor = '#ced4da';
                                                            }
                                                        }}
                                                    >
                                                        {entity.text}
                                                        {entity.score && (
                                                            <span style={{ 
                                                                fontSize: '11px', 
                                                                marginLeft: '6px',
                                                                opacity: 0.7
                                                            }}>
                                                                ({(entity.score * 100).toFixed(0)}%)
                                                            </span>
                                                        )}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {bioGPTResponse && (
                                <div style={{ 
                                    padding: '16px', 
                                    backgroundColor: '#e7f5ff', 
                                    borderRadius: '2px', 
                                    border: '1px solid #339af0',
                                    borderLeft: '4px solid #339af0'
                                }}>
                                    <h4 style={{ 
                                        margin: '0 0 10px 0', 
                                        fontSize: '15px', 
                                        fontWeight: '600',
                                        color: '#1864ab'
                                    }}>Entity Information</h4>
                                    <p style={{ 
                                        margin: 0,
                                        fontSize: '14px',
                                        lineHeight: '1.7',
                                        color: '#495057'
                                    }}>{bioGPTResponse}</p>
                                </div>
                            )}

                            {!processedText && Object.keys(entitiesByCategory).length === 0 && meshTerms.length === 0 && !bioGPTResponse && (
                                <div style={{ 
                                    textAlign: 'center', 
                                    padding: '60px 20px',
                                    color: '#adb5bd'
                                }}>
                                    <p style={{ fontSize: '14px', margin: 0, lineHeight: '1.6' }}>
                                        Upload a scientific article or enter text, then click one of the buttons to begin analysis:
                                    </p>
                                    <ul style={{ fontSize: '13px', textAlign: 'left', display: 'inline-block', marginTop: '12px', color: '#6c757d' }}>
                                        <li><strong>Analyze Text:</strong> BioGPT text generation</li>
                                        <li><strong>Extract Entities:</strong> Biomedical NER (diseases, chemicals, genes)</li>
                                        <li><strong>Extract MeSH Terms:</strong> Fine-tuned LLaMA 3.1-8B for MeSH terminology</li>
                                    </ul>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default BioGPTComponent;