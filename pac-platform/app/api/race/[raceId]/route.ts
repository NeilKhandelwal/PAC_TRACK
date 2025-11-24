import { NextResponse } from 'next/server';
import { db } from '../../../../lib/firebase';
import { doc, getDoc } from 'firebase/firestore';

export async function GET(request: Request, { params }: { params: Promise<{ raceId: string }> }) {
    const { raceId } = await params;
    try {
        const docRef = doc(db, 'races', raceId);
        const snap = await getDoc(docRef);
        if (!snap.exists()) {
            return NextResponse.json({ error: 'Race not found' }, { status: 404 });
        }
        const data = snap.data();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Error fetching race data:', error);
        return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
    }
}
