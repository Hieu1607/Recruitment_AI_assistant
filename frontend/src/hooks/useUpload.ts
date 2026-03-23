import { useState } from 'react';

export const useUpload = () => {
    const [isUploading, setIsUploading] = useState(false);
    
    const upload = async (file: File) => {
        setIsUploading(true);
        // ... upload logic ...
        setIsUploading(false);
    };

    return { upload, isUploading };
};
