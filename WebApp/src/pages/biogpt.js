import React from 'react';
import BioGPTComponent from '../components/BioGPTComponent';

const BioGPT = () => {
    return (
        <div
            style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                minHeight: '100vh',
                padding: '20px',
                boxSizing: 'border-box'
            }}
        >
            <BioGPTComponent />
        </div>
    );
};

export default BioGPT;