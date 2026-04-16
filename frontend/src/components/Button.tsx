import type { ReactNode } from 'react';

export const Button = ({ children, onClick }: { children: ReactNode, onClick?: () => void }) => {
    return (
        <button onClick={onClick} style={{ padding: '8px 16px', borderRadius: '4px' }}>
            {children}
        </button>
    );
};
