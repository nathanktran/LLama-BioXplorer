import React, { useState } from 'react';
import axios from 'axios';
import ScrollableBox from './ScrollableBox';

const BioGPTComponent = () => {
    const [inputText, setInputText] = useState('');
    const [processedText, setProcessedText] = useState('');
    const [entities, setEntities] = useState([]);
    const [loading, setLoading] = useState(false);
    const [selectedEntity, setSelectedEntity] = useState(null);
    const [bioGPTResponse, setBioGPTResponse] = useState('');

    const handleTextChange = (event) => {
        setInputText(event.target.value);
        setProcessedText('');
        setEntities([]);
        setBioGPTResponse('');
    };

    const handleProcess = async () => {
        if (!inputText) {
            return;
        }

        setLoading(true);
        try {
            // Replace with your actual BioGPT API endpoint
            const response = await axios.post('http://localhost:5003/process_biogpt', {
                text: inputText
            });

            setProcessedText(response.data.processed_text);
            setEntities(response.data.entities || []);
        } catch (error) {
            console.error('Processing failed:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleEntityClick = async (entity) => {
        setSelectedEntity(entity);
        try {
            // Replace with your actual entity query endpoint
            const response = await axios.post('http://localhost:5003/query_entity', {
                entity: entity
            });
            setBioGPTResponse(response.data.response);
        } catch (error) {
            console.error('Entity query failed:', error);
        }
    };

    return (
        <div
            style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                width: '100%',
                padding: '20px',
                boxSizing: 'border-box',
                fontFamily: 'Arial, sans-serif',
            }}
        >
            <div
                style={{
                    backgroundColor: '#b4d3b2',
                    width: '100%',
                    height: '70px',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    borderWidth: '2px',
                    borderStyle: 'solid',
                    borderColor: '#b4d3b2',
                    borderRadius: '10px',
                    marginBottom: '20px',
                    boxSizing: 'border-box'
                }}
            >
                <h2 style={{ color: 'black', margin: 0 }}>BioGPT Analysis with Entity Linkage</h2>
            </div>

            <div
                style={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'flex-start',
                    height: '80vh',
                    flexDirection: 'row',
                    gap: '20px',
                    width: '100%',
                }}
            >
                {/* Left Box - Input and Controls */}
                <div
                    style={{
                        display: 'flex',
                        flexDirection: 'column',
                        backgroundColor: '#b4d3b2',
                        width: '30%',
                        height: '100%',
                        padding: '20px',
                        borderWidth: '2px',
                        borderRadius: '10px',
                        borderColor: '#b4d3b2',
                        borderStyle: 'solid',
                        boxSizing: 'border-box',
                        position: 'relative'
                    }}
                >
                    <textarea
                        style={{
                            width: '100%',
                            height: '200px',
                            marginBottom: '20px',
                            padding: '10px',
                            borderRadius: '5px',
                            border: '1px solid #ccc',
                            resize: 'vertical'
                        }}
                        placeholder="Enter your text here..."
                        value={inputText}
                        onChange={handleTextChange}
                    />
                    <div style={{ position: 'absolute', bottom: '20px', width: 'calc(100% - 40px)' }}>
                        <button
                            className="button-27"
                            onClick={handleProcess}
                            disabled={!inputText || loading}
                        >
                            Analyze with BioGPT
                        </button>
                    </div>
                </div>

                {/* Right Box - Results */}
                <div
                    style={{
                        display: 'flex',
                        flexDirection: 'column',
                        backgroundColor: '#b4d3b2',
                        width: '70%',
                        height: '100%',
                        padding: '20px',
                        borderWidth: '2px',
                        borderRadius: '10px',
                        borderColor: '#b4d3b2',
                        borderStyle: 'solid',
                        boxSizing: 'border-box',
                        overflowY: 'auto'
                    }}
                >
                    {loading ? (
                        <p>Processing text...</p>
                    ) : (
                        <>
                            {processedText && (
                                <div
                                    style={{
                                        border: '1px solid #696969',
                                        padding: '10px',
                                        borderRadius: '10px',
                                        marginBottom: '20px',
                                        backgroundColor: 'white'
                                    }}
                                >
                                    <ScrollableBox title={processedText} />
                                </div>
                            )}

                            {entities.length > 0 && (
                                <div style={{ marginTop: '20px' }}>
                                    <h4>Identified Entities:</h4>
                                    <div style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: '10px' }}>
                                        {entities.map((entity, index) => (
                                            <div
                                                key={index}
                                                onClick={() => handleEntityClick(entity)}
                                                style={{
                                                    backgroundColor: selectedEntity === entity ? '#557153' : '#69a765',
                                                    color: selectedEntity === entity ? 'white' : 'black',
                                                    padding: '10px 15px',
                                                    borderRadius: '20px',
                                                    fontWeight: 'bold',
                                                    cursor: 'pointer',
                                                    textAlign: 'center',
                                                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
                                                }}
                                            >
                                                {entity}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {bioGPTResponse && (
                                <div
                                    style={{
                                        marginTop: '20px',
                                        padding: '15px',
                                        backgroundColor: 'white',
                                        borderRadius: '10px',
                                        border: '1px solid #696969'
                                    }}
                                >
                                    <h4>Entity Information:</h4>
                                    <p>{bioGPTResponse}</p>
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