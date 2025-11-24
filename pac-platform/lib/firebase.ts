import { initializeApp, getApps } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
    apiKey: '<YOUR-API-KEY>',
    authDomain: '<your>.firebaseapp.com',
    projectId: '<your-project-id>',
    // add other config fields as needed
};

export const app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig);
export const db = getFirestore(app);
